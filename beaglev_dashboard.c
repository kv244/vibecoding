/*
 * beaglev_dashboard.c
 *
 * ncurses system monitor + CVE scanner for BeagleV-Fire (RISC-V / PolarFire SoC)
 *
 * Build (native on BeagleV):
 *   gcc -O2 -march=rv64gc -o beaglev_dashboard beaglev_dashboard.c \
 *       -lncurses -lsqlite3 -lpthread
 *
 * Build (cross from x86 host):
 *   riscv64-linux-gnu-gcc -O2 -march=rv64gc \
 *       --sysroot=./riscv-sysroot \
 *       -o beaglev_dashboard beaglev_dashboard.c \
 *       -lncurses -lsqlite3 -lpthread
 *
 * CVE database schema (auto-created by init_cve_db() on first run):
 *
 *   packages(id, name, version)        -- populated from /var/lib/dpkg/status
 *   cves(id, cve_id, package,          -- populated externally (NVD feeds etc.)
 *        fixed_ver, severity, description)
 *
 * Severity values expected: CRITICAL / HIGH / MEDIUM / LOW
 *
 * Key bindings:
 *   q / Q    quit
 *   r / R    trigger a fresh CVE rescan
 *   j / DOWN scroll CVE list down
 *   k / UP   scroll CVE list up
 */

#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sqlite3.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/utsname.h>
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#include <sys/select.h>

/* ------------------------------------------------------------------ */
/* Configuration                                                        */
/* ------------------------------------------------------------------ */

#define CVE_DB         "cve_database.db"   /* SQLite CVE store         */
#define DPKG_STATUS    "/var/lib/dpkg/status"
#define PI5_IP         "192.168.1.35"      /* Remote Raspberry Pi 5    */
#define PI5_PORT       22                  /* SSH port used as probe   */
#define NET_TIMEOUT_MS 300                 /* TCP probe timeout (ms)   */

/* ------------------------------------------------------------------ */
/* ncurses color pair IDs                                               */
/* ------------------------------------------------------------------ */

#define CP_DEFAULT  1   /* white on black  */
#define CP_ALERT    2   /* red   on black  */
#define CP_HEADER   3   /* cyan  on black  */
#define CP_OK       4   /* green on black  */
#define CP_WARN     5   /* yellow on black */

/* ------------------------------------------------------------------ */
/* CVE result: singly-linked list node                                  */
/* ------------------------------------------------------------------ */

typedef struct CVEResult {
    char cve_id[32];
    char package[128];
    char installed_ver[64];
    char fixed_ver[64];
    char severity[16];
    char description[256];
    struct CVEResult *next;
} CVEResult;

/* ------------------------------------------------------------------ */
/* Shared scanner state (scanner thread writes, UI thread reads)        */
/* All fields protected by lock except scan_done which is read-only     */
/* after the thread signals completion.                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    pthread_mutex_t lock;
    CVEResult      *head;           /* head of findings list           */
    int             count;          /* total findings                  */
    int             scan_done;      /* 0 = running, 1 = complete       */
    char            status_msg[64]; /* human-readable progress string  */
} ScanState;

static ScanState g_scan = {
    .lock       = PTHREAD_MUTEX_INITIALIZER,
    .head       = NULL,
    .count      = 0,
    .scan_done  = 0,
    .status_msg = "Idle"
};

/* ================================================================== */
/* Version string comparison                                            */
/*                                                                      */
/* Returns -1 / 0 / 1 (like strcmp) for dpkg-style version strings.    */
/* Handles purely numeric segments; non-digit separators are compared   */
/* lexically. Used to determine if an installed package version is      */
/* older than a CVE's fixed_ver.                                        */
/* ================================================================== */

/*
 * parse_digits() -- RISC-V inline ASM digit accumulation
 *
 * Reads one run of ASCII decimal digits from *pp, advances *pp past
 * them, and returns the integer value.
 *
 * The accumulation loop uses the shift-add multiply trick to compute
 * acc*10 without the M-extension mul instruction:
 *
 *   acc * 10 = (acc << 3) + (acc << 1)
 *
 * This keeps the code correct on cores where M is absent, while still
 * being a single tight sequence on U54 cores that do have M.
 *
 * ASM operand roles:
 *   [acc]  "+r"  -- accumulator, read and written each iteration
 *   [p]    "+r"  -- byte pointer, advanced by 1 after each digit
 *   [tmp]  "=&r" -- scratch register (early-clobber)
 */
static inline long parse_digits(const char **pp)
{
    const char *p = *pp;
    long acc = 0;

    while ((unsigned)(*p - '0') <= 9u) {
        long tmp;
        __asm__ volatile (
            "slli   %[tmp], %[acc], 1\n\t"      /* tmp = acc * 2           */
            "slli   %[acc], %[acc], 3\n\t"      /* acc = acc * 8           */
            "add    %[acc], %[acc], %[tmp]\n\t" /* acc = acc*8 + acc*2     */
            "lbu    %[tmp], 0(%[p])\n\t"        /* tmp = *p (zero-extend)  */
            "addi   %[tmp], %[tmp], -48\n\t"    /* tmp -= '0'              */
            "add    %[acc], %[acc], %[tmp]\n\t" /* acc += digit            */
            "addi   %[p],   %[p],   1\n\t"      /* advance pointer         */
            : [acc] "+r" (acc),
              [p]   "+r" (p),
              [tmp] "=&r" (tmp)
            :
            : /* no additional clobbers */
        );
    }

    *pp = p;
    return acc;
}

static int ver_cmp_numeric(const char *a, const char *b) {
    while (*a || *b) {
        /* Skip non-digit characters, comparing them lexically */
        while (*a && !((unsigned)(*a - '0') <= 9u)) {
            if (!*b || *a < *b) return -1;
            if (*a > *b)        return  1;
            a++; b++;
        }
        /* Parse and compare numeric segments via ASM helper */
        long na = parse_digits(&a);
        long nb = parse_digits(&b);
        if (na < nb) return -1;
        if (na > nb) return  1;
    }
    return 0;
}

/* ================================================================== */
/* SQLite helpers                                                       */
/* ================================================================== */

/*
 * init_cve_db() -- create schema if it does not already exist.
 * Safe to call on an already-initialised database.
 */
static int init_cve_db(sqlite3 *db) {
    const char *sql =
        "CREATE TABLE IF NOT EXISTS packages ("
        "  id      INTEGER PRIMARY KEY,"
        "  name    TEXT NOT NULL,"
        "  version TEXT NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS cves ("
        "  id          INTEGER PRIMARY KEY,"
        "  cve_id      TEXT NOT NULL,"
        "  package     TEXT NOT NULL,"
        "  fixed_ver   TEXT,"          /* NULL means no fix available   */
        "  severity    TEXT,"
        "  description TEXT"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_cve_package ON cves(package);";

    char *errmsg = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "init_cve_db: %s\n", errmsg);
        sqlite3_free(errmsg);
        return -1;
    }
    return 0;
}

/*
 * import_dpkg_packages() -- parse /var/lib/dpkg/status and bulk-insert
 * all installed packages into the packages table.
 *
 * Clears existing rows first so stale uninstalled packages do not linger.
 * Wrapped in a single BEGIN/COMMIT transaction for performance.
 */
static int import_dpkg_packages(sqlite3 *db) {
    FILE *fp = fopen(DPKG_STATUS, "r");
    if (!fp) {
        perror("fopen dpkg/status");
        return -1;
    }

    sqlite3_exec(db, "DELETE FROM packages;", NULL, NULL, NULL);

    sqlite3_stmt *stmt = NULL;
    const char *insert_sql =
        "INSERT INTO packages (name, version) VALUES (?, ?);";
    if (sqlite3_prepare_v2(db, insert_sql, -1, &stmt, NULL) != SQLITE_OK) {
        fclose(fp);
        return -1;
    }

    char  line[512];
    char  pkg_name[128]    = {0};
    char  pkg_version[128] = {0};
    int   installed        = 0;

    sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);

    while (fgets(line, sizeof(line), fp)) {
        line[strcspn(line, "\n")] = '\0';   /* strip trailing newline  */

        if (strncmp(line, "Package: ", 9) == 0) {
            strncpy(pkg_name, line + 9, sizeof(pkg_name) - 1);
            pkg_version[0] = '\0';
            installed = 0;

        } else if (strncmp(line, "Version: ", 9) == 0) {
            strncpy(pkg_version, line + 9, sizeof(pkg_version) - 1);

        } else if (strncmp(line, "Status: ", 8) == 0) {
            installed = (strstr(line, "installed") != NULL);

        } else if (line[0] == '\0') {
            /* Blank line marks end of a stanza */
            if (installed && pkg_name[0] && pkg_version[0]) {
                sqlite3_reset(stmt);
                sqlite3_bind_text(stmt, 1, pkg_name,    -1, SQLITE_TRANSIENT);
                sqlite3_bind_text(stmt, 2, pkg_version, -1, SQLITE_TRANSIENT);
                sqlite3_step(stmt);
            }
            pkg_name[0]    = '\0';
            pkg_version[0] = '\0';
            installed      = 0;
        }
    }

    /* Handle the final stanza if the file does not end with a blank line */
    if (installed && pkg_name[0] && pkg_version[0]) {
        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, pkg_name,    -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, pkg_version, -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
    }

    sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
    sqlite3_finalize(stmt);
    fclose(fp);
    return 0;
}

/*
 * run_cve_scan() -- cross-reference installed packages against the CVE
 * table and build the g_scan linked list with matching findings.
 *
 * A CVE is flagged only when:
 *   - fixed_ver IS NULL (no fix exists yet), OR
 *   - installed_ver < fixed_ver (still running a vulnerable version)
 *
 * Results are inserted at the head of g_scan.head in SQL ORDER BY
 * severity (CRITICAL first) so the list is pre-sorted for display.
 */
static void run_cve_scan(sqlite3 *db) {
    const char *sql =
        "SELECT c.cve_id, c.package, p.version, c.fixed_ver,"
        "       c.severity, c.description "
        "FROM cves c "
        "JOIN packages p ON p.name = c.package "
        "ORDER BY "
        "  CASE c.severity "
        "    WHEN 'CRITICAL' THEN 1 "
        "    WHEN 'HIGH'     THEN 2 "
        "    WHEN 'MEDIUM'   THEN 3 "
        "    ELSE 4 "
        "  END;";

    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        pthread_mutex_lock(&g_scan.lock);
        snprintf(g_scan.status_msg, sizeof(g_scan.status_msg),
                 "CVE query failed: %s", sqlite3_errmsg(db));
        pthread_mutex_unlock(&g_scan.lock);
        return;
    }

    /* Free any results from a previous scan run */
    pthread_mutex_lock(&g_scan.lock);
    CVEResult *cur = g_scan.head;
    while (cur) { CVEResult *tmp = cur->next; free(cur); cur = tmp; }
    g_scan.head  = NULL;
    g_scan.count = 0;
    pthread_mutex_unlock(&g_scan.lock);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *cve_id      = (const char *)sqlite3_column_text(stmt, 0);
        const char *package     = (const char *)sqlite3_column_text(stmt, 1);
        const char *inst_ver    = (const char *)sqlite3_column_text(stmt, 2);
        const char *fixed_ver   = (const char *)sqlite3_column_text(stmt, 3);
        const char *severity    = (const char *)sqlite3_column_text(stmt, 4);
        const char *description = (const char *)sqlite3_column_text(stmt, 5);

        /* Skip if a fix exists and the installed version already meets it */
        if (fixed_ver && inst_ver) {
            if (ver_cmp_numeric(inst_ver, fixed_ver) >= 0) continue;
        }

        CVEResult *res = calloc(1, sizeof(CVEResult));
        if (!res) continue;

        strncpy(res->cve_id,        cve_id      ? cve_id      : "N/A", sizeof(res->cve_id)        - 1);
        strncpy(res->package,       package     ? package     : "N/A", sizeof(res->package)       - 1);
        strncpy(res->installed_ver, inst_ver    ? inst_ver    : "N/A", sizeof(res->installed_ver) - 1);
        strncpy(res->fixed_ver,     fixed_ver   ? fixed_ver   : "N/A", sizeof(res->fixed_ver)     - 1);
        strncpy(res->severity,      severity    ? severity    : "N/A", sizeof(res->severity)      - 1);
        strncpy(res->description,   description ? description : "N/A", sizeof(res->description)   - 1);

        pthread_mutex_lock(&g_scan.lock);
        res->next    = g_scan.head;
        g_scan.head  = res;
        g_scan.count++;
        pthread_mutex_unlock(&g_scan.lock);
    }

    sqlite3_finalize(stmt);
}

/*
 * scanner_thread() -- background worker that drives the full scan pipeline:
 *   1. Open SQLite DB (WAL mode so UI reads do not block)
 *   2. Ensure schema exists
 *   3. Import current dpkg package list
 *   4. Run CVE cross-reference
 *   5. Signal completion via g_scan.scan_done
 */
static void *scanner_thread(void *arg) {
    (void)arg;
    sqlite3 *db = NULL;

    pthread_mutex_lock(&g_scan.lock);
    snprintf(g_scan.status_msg, sizeof(g_scan.status_msg), "Opening DB...");
    g_scan.scan_done = 0;
    pthread_mutex_unlock(&g_scan.lock);

    if (sqlite3_open(CVE_DB, &db) != SQLITE_OK) {
        pthread_mutex_lock(&g_scan.lock);
        snprintf(g_scan.status_msg, sizeof(g_scan.status_msg),
                 "DB open failed: %s", sqlite3_errmsg(db));
        g_scan.scan_done = 1;
        pthread_mutex_unlock(&g_scan.lock);
        return NULL;
    }

    /* WAL mode: allows concurrent reads while we write package data */
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);

    init_cve_db(db);

    pthread_mutex_lock(&g_scan.lock);
    snprintf(g_scan.status_msg, sizeof(g_scan.status_msg), "Importing dpkg...");
    pthread_mutex_unlock(&g_scan.lock);

    import_dpkg_packages(db);

    pthread_mutex_lock(&g_scan.lock);
    snprintf(g_scan.status_msg, sizeof(g_scan.status_msg), "Scanning CVEs...");
    pthread_mutex_unlock(&g_scan.lock);

    run_cve_scan(db);

    sqlite3_close(db);

    pthread_mutex_lock(&g_scan.lock);
    snprintf(g_scan.status_msg, sizeof(g_scan.status_msg),
             "Scan complete: %d finding(s)", g_scan.count);
    g_scan.scan_done = 1;
    pthread_mutex_unlock(&g_scan.lock);

    return NULL;
}

/* ================================================================== */
/* ncurses draw helpers                                                  */
/* ================================================================== */

static void draw_borders(WINDOW *win, const char *title) {
    box(win, 0, 0);
    wattron(win, COLOR_PAIR(CP_HEADER) | A_BOLD);
    mvwprintw(win, 0, 2, " %s ", title);
    wattroff(win, COLOR_PAIR(CP_HEADER) | A_BOLD);
}

/*
 * update_system_info() -- render the left-hand health panel.
 *
 * Shows: CPU temp (thermal_zone0), 1-min load average, thread count,
 * root filesystem usage, and uptime.
 *
 * Uptime h/m split uses RISC-V divu/remu directly. The U54 application
 * cores on the PolarFire SoC implement the M extension so these are
 * single-cycle. Using inline ASM guarantees the instruction even at -O0.
 */
static void update_system_info(WINDOW *win) {
    struct sysinfo si;
    struct statvfs vfs;
    double load[1];

    sysinfo(&si);
    if (getloadavg(load, 1) != 1) load[0] = -1.0;

    werase(win);
    draw_borders(win, "BEAGLEV HEALTH");

    /* Thermal zone 0 -- BeagleV / PolarFire TMC reports millidegrees C */
    float temp = -1.0f;
    FILE *tp = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    if (tp) {
        int raw = 0;
        if (fscanf(tp, "%d", &raw) == 1) temp = raw / 1000.0f;
        fclose(tp);
    }

    int row = 2;
    if (temp >= 0.0f)
        mvwprintw(win, row++, 2, "Temp:    %.1f C", temp);
    else
        mvwprintw(win, row++, 2, "Temp:    N/A");

    if (load[0] >= 0.0)
        mvwprintw(win, row++, 2, "Load:    %.2f", load[0]);
    else
        mvwprintw(win, row++, 2, "Load:    N/A");

    mvwprintw(win, row++, 2, "Threads: %u", si.procs);

    if (statvfs("/", &vfs) == 0) {
        double pct = 100.0 * (1.0 - (double)vfs.f_bfree / (double)vfs.f_blocks);
        mvwprintw(win, row++, 2, "Disk:    %.1f%%", pct);
    }

    /* Uptime split using RISC-V M-extension divide/remainder */
    {
        unsigned long up_h, up_m, tmp;
        unsigned long uptime = (unsigned long)si.uptime;
        __asm__ (
            "li     %[tmp], 3600\n\t"
            "divu   %[h],   %[up], %[tmp]\n\t"  /* hours = uptime / 3600  */
            "remu   %[m],   %[up], %[tmp]\n\t"  /* rem   = uptime % 3600  */
            "li     %[tmp], 60\n\t"
            "divu   %[m],   %[m],  %[tmp]\n\t"  /* mins  = rem / 60       */
            : [h]   "=&r" (up_h),
              [m]   "=&r" (up_m),
              [tmp] "=&r" (tmp)
            : [up]  "r"   (uptime)
            :
        );
        mvwprintw(win, row++, 2, "Uptime:  %luh %lum", up_h, up_m);
    }

    wrefresh(win);
}

/* ------------------------------------------------------------------ */
/* sysfs helper: read one line from path into buf, strip newline.      */
/* Returns 0 on success, -1 if the file cannot be opened or read.     */
/* ------------------------------------------------------------------ */
static int read_sysfs(const char *path, char *buf, size_t len) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, (int)len, f)) { fclose(f); buf[0] = '\0'; return -1; }
    fclose(f);
    buf[strcspn(buf, "\n")] = '\0';
    return 0;
}

/*
 * FIC bridge table -- PolarFire SoC exposes up to four AXI bridges via
 * the fpga-bridge driver. Each entry maps a sysfs state file to a short
 * label and the fabric address window it controls.
 *
 * Bridge states and what they mean for this board:
 *
 *   br0 (FIC0) -- AXI master to 0x60000000 fabric window.
 *                 DISABLED here is the root cause of 0x60000000 bus faults.
 *   br1 (FIC1) -- AXI master to 0x70000000 fabric window.
 *   br2 (FIC2) -- AXI-Lite config bridge (register access to fabric IP).
 *   br3 (FIC3) -- DMA bridge (if instantiated in the bitstream).
 *
 * A "not in DT" result means the bridge binding is absent from the
 * Device Tree overlay -- distinct from disabled, because it cannot be
 * enabled at runtime without a DT fix.
 */
#define FIC_BRIDGE_COUNT 4

static const struct {
    const char *sysfs_state;
    const char *label;
    const char *addr_window;
} fic_bridges[FIC_BRIDGE_COUNT] = {
    { "/sys/class/fpga_bridge/br0/state", "FIC0", "0x60000000" },
    { "/sys/class/fpga_bridge/br1/state", "FIC1", "0x70000000" },
    { "/sys/class/fpga_bridge/br2/state", "FIC2", "AXI-Lite"   },
    { "/sys/class/fpga_bridge/br3/state", "FIC3", "DMA"        },
};

/*
 * update_fabric_info() -- render the FPGA manager state, all four FIC
 * bridge enable bits, and the non-blocking TCP probe to the Pi 5.
 *
 * The FPGA manager state alone is insufficient to diagnose bus faults:
 * the fabric can be "operating" while a bridge is still gated. Showing
 * each bridge state gives an immediate visual on which AXI window is
 * causing the fault without needing to attach a debugger.
 *
 * Network probe uses O_NONBLOCK + select() so a dead host never blocks
 * the UI thread. SO_SNDTIMEO is not used because it does not reliably
 * affect connect() on Linux -- select() is the correct mechanism.
 */
static void update_fabric_info(WINDOW *win) {
    werase(win);
    draw_borders(win, "FPGA & NETWORK");

    /* FPGA manager bitstream state */
    int row = 2;
    mvwprintw(win, row, 2, "Fabric:  ");
    FILE *fs = fopen("/sys/class/fpga_manager/fpga0/state", "r");
    if (fs) {
        char state[32] = {0};
        fgets(state, sizeof(state), fs);
        fclose(fs);
        state[strcspn(state, "\n")] = '\0';

        if (strstr(state, "operating")) {
            wattron(win, COLOR_PAIR(CP_OK) | A_BOLD);
            wprintw(win, "ACTIVE");
            wattroff(win, COLOR_PAIR(CP_OK) | A_BOLD);
        } else {
            wattron(win, COLOR_PAIR(CP_ALERT) | A_BOLD);
            wprintw(win, "%-16s", state);
            wattroff(win, COLOR_PAIR(CP_ALERT) | A_BOLD);
        }
    } else {
        wattron(win, COLOR_PAIR(CP_WARN));
        wprintw(win, "unavailable");
        wattroff(win, COLOR_PAIR(CP_WARN));
    }
    row++;

    /* FIC bridge enable bits */
    mvwprintw(win, row++, 2, "FIC Bridges:");

    for (int i = 0; i < FIC_BRIDGE_COUNT; i++) {
        char state[32] = {0};
        int  exists    = (access(fic_bridges[i].sysfs_state, F_OK) == 0);

        mvwprintw(win, row, 4, "%-5s (%s): ",
                  fic_bridges[i].label,
                  fic_bridges[i].addr_window);

        if (!exists) {
            /* No sysfs node -- bridge is absent from Device Tree */
            wattron(win, COLOR_PAIR(CP_WARN));
            wprintw(win, "not in DT");
            wattroff(win, COLOR_PAIR(CP_WARN));
            row++;
            continue;
        }

        if (read_sysfs(fic_bridges[i].sysfs_state, state, sizeof(state)) < 0) {
            wattron(win, COLOR_PAIR(CP_WARN));
            wprintw(win, "read err");
            wattroff(win, COLOR_PAIR(CP_WARN));
            row++;
            continue;
        }

        if (strcmp(state, "enabled") == 0) {
            wattron(win, COLOR_PAIR(CP_OK) | A_BOLD);
            wprintw(win, "ENABLED");
            wattroff(win, COLOR_PAIR(CP_OK) | A_BOLD);
        } else {
            /* Bridge gated -- accesses to this window will bus-fault */
            wattron(win, COLOR_PAIR(CP_ALERT) | A_BOLD);
            wprintw(win, "DISABLED  <-- bus fault risk");
            wattroff(win, COLOR_PAIR(CP_ALERT) | A_BOLD);
        }
        row++;
    }

    row++; /* blank line before network section */

    /* Non-blocking TCP probe to Raspberry Pi 5 */
    mvwprintw(win, row, 2, "Pi 5 (%s):  ", PI5_IP);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        wattron(win, COLOR_PAIR(CP_ALERT));
        wprintw(win, "socket err");
        wattroff(win, COLOR_PAIR(CP_ALERT));
    } else {
        /* Set non-blocking before connect so we can poll with select() */
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(PI5_PORT);
        inet_pton(AF_INET, PI5_IP, &addr.sin_addr);

        /* connect() returns immediately with EINPROGRESS for non-blocking */
        connect(sock, (struct sockaddr *)&addr, sizeof(addr));

        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(sock, &wfds);

        struct timeval tv;
        tv.tv_sec  = 0;
        tv.tv_usec = NET_TIMEOUT_MS * 1000;

        int ready = select(sock + 1, NULL, &wfds, NULL, &tv);
        if (ready > 0) {
            /* select() fired -- check SO_ERROR to catch ECONNREFUSED etc. */
            int err = 0;
            socklen_t elen = sizeof(err);
            getsockopt(sock, SOL_SOCKET, SO_ERROR, &err, &elen);
            if (err == 0) {
                wattron(win, COLOR_PAIR(CP_OK) | A_BOLD);
                wprintw(win, "ONLINE");
                wattroff(win, COLOR_PAIR(CP_OK) | A_BOLD);
            } else {
                wattron(win, COLOR_PAIR(CP_ALERT));
                wprintw(win, "REFUSED (%s)", strerror(err));
                wattroff(win, COLOR_PAIR(CP_ALERT));
            }
        } else {
            wattron(win, COLOR_PAIR(CP_ALERT));
            wprintw(win, "OFFLINE");
            wattroff(win, COLOR_PAIR(CP_ALERT));
        }
        close(sock);
    }

    wrefresh(win);
}

/*
 * update_cve_log() -- render the CVE findings panel.
 *
 * Each entry uses two rows: the first shows CVE ID, severity, package
 * name, installed version and fixed version; the second shows a
 * truncated description. scroll_offset skips that many entries from the
 * head of the list (already sorted CRITICAL-first by the SQL query).
 *
 * Severity -> color mapping uses a branchless RISC-V inline ASM block
 * that loads the first byte of the severity string and computes a rank
 * integer (0-3) via seqz + arithmetic, avoiding four strcmp calls per
 * row on the hot render path.
 *
 *   Rank formula:  rank = 3 - 3*is_C - 2*is_H - is_M
 *   where is_X = 1 if severity[0] == X, else 0.
 *
 *   rank 0 (CRITICAL) -> CP_ALERT (red)
 *   rank 1 (HIGH)     -> CP_ALERT (red)
 *   rank 2 (MEDIUM)   -> CP_WARN  (yellow)
 *   rank 3 (LOW/other)-> CP_OK    (green)
 */
static void update_cve_log(WINDOW *win, int scroll_offset) {
    werase(win);
    draw_borders(win, "BEAGLEV CVE SCANNER");

    int max_y, max_x;
    getmaxyx(win, max_y, max_x);
    (void)max_x;
    int content_rows = max_y - 3;   /* rows available below the status bar */

    /* Snapshot shared state under lock */
    pthread_mutex_lock(&g_scan.lock);
    char status[64];
    strncpy(status, g_scan.status_msg, sizeof(status) - 1);
    int total = g_scan.count;
    int done  = g_scan.scan_done;
    pthread_mutex_unlock(&g_scan.lock);

    /* Status bar */
    if (!done) {
        wattron(win, COLOR_PAIR(CP_WARN) | A_BOLD);
        mvwprintw(win, 1, 2, "[*] %s", status);
        wattroff(win, COLOR_PAIR(CP_WARN) | A_BOLD);
    } else if (total == 0) {
        wattron(win, COLOR_PAIR(CP_OK) | A_BOLD);
        mvwprintw(win, 1, 2, "[+] No CVEs found for installed packages.");
        wattroff(win, COLOR_PAIR(CP_OK) | A_BOLD);
    } else {
        wattron(win, COLOR_PAIR(CP_ALERT) | A_BOLD);
        mvwprintw(win, 1, 2, "[!] %s  (scroll: j/k)", status);
        wattroff(win, COLOR_PAIR(CP_ALERT) | A_BOLD);
    }

    pthread_mutex_lock(&g_scan.lock);
    CVEResult *cur = g_scan.head;
    int idx = 0;

    /* Advance list pointer to the current scroll position */
    while (cur && idx < scroll_offset) { cur = cur->next; idx++; }

    int row = 2;
    while (cur && row < 2 + content_rows) {

        /*
         * Branchless severity -> color rank via RISC-V inline ASM.
         * Load severity[0], compute three boolean flags with seqz,
         * then fold them into rank = 3 - 3*is_C - 2*is_H - is_M.
         */
        int cp;
        {
            long byte, r_C, r_H, r_M, rank;
            __asm__ (
                "lbu    %[byte], 0(%[sev])\n\t"      /* load severity[0]        */
                "addi   %[r_C], %[byte], -67\n\t"    /* byte - 'C'              */
                "seqz   %[r_C], %[r_C]\n\t"          /* 1 if CRITICAL, else 0   */
                "addi   %[r_H], %[byte], -72\n\t"    /* byte - 'H'              */
                "seqz   %[r_H], %[r_H]\n\t"          /* 1 if HIGH, else 0       */
                "addi   %[r_M], %[byte], -77\n\t"    /* byte - 'M'              */
                "seqz   %[r_M], %[r_M]\n\t"          /* 1 if MEDIUM, else 0     */
                "li     %[rank], 3\n\t"               /* rank = 3 - 3*C - 2*H - M */
                "sub    %[rank], %[rank], %[r_C]\n\t"
                "sub    %[rank], %[rank], %[r_C]\n\t"
                "sub    %[rank], %[rank], %[r_C]\n\t"
                "sub    %[rank], %[rank], %[r_H]\n\t"
                "sub    %[rank], %[rank], %[r_H]\n\t"
                "sub    %[rank], %[rank], %[r_M]\n\t"
                : [byte] "=&r" (byte),
                  [r_C]  "=&r" (r_C),
                  [r_H]  "=&r" (r_H),
                  [r_M]  "=&r" (r_M),
                  [rank] "=&r" (rank)
                : [sev] "r" (cur->severity)
                :
            );

            static const int cp_table[4] = {
                CP_ALERT,   /* rank 0 = CRITICAL */
                CP_ALERT,   /* rank 1 = HIGH     */
                CP_WARN,    /* rank 2 = MEDIUM   */
                CP_OK       /* rank 3 = LOW/other*/
            };
            if ((unsigned long)rank > 3) rank = 3;   /* clamp against bad input */
            cp = cp_table[rank];
        }

        wattron(win, COLOR_PAIR(cp) | A_BOLD);
        mvwprintw(win, row, 2, "%-10s %-8s", cur->cve_id, cur->severity);
        wattroff(win, COLOR_PAIR(cp) | A_BOLD);

        wprintw(win, " %-20s installed:%-12s fixed:%-12s",
                cur->package, cur->installed_ver, cur->fixed_ver);
        row++;

        /* Description row -- truncated to fit the window width */
        if (row < 2 + content_rows) {
            mvwprintw(win, row, 4, "%.70s", cur->description);
            row++;
        }

        cur = cur->next;
    }
    pthread_mutex_unlock(&g_scan.lock);

    wrefresh(win);
}

/* ================================================================== */
/* main                                                                  */
/* ================================================================== */

int main(void) {
    /* Start CVE scanner in background immediately so it runs while     */
    /* ncurses initialises and the first UI frame renders.              */
    pthread_t scan_tid;
    pthread_create(&scan_tid, NULL, scanner_thread, NULL);

    /* ncurses setup */
    initscr();
    start_color();
    use_default_colors();
    cbreak();
    noecho();
    curs_set(0);
    keypad(stdscr, TRUE);
    timeout(1500);   /* refresh every 1.5 s */

    init_pair(CP_DEFAULT, COLOR_WHITE,  COLOR_BLACK);
    init_pair(CP_ALERT,   COLOR_RED,    COLOR_BLACK);
    init_pair(CP_HEADER,  COLOR_CYAN,   COLOR_BLACK);
    init_pair(CP_OK,      COLOR_GREEN,  COLOR_BLACK);
    init_pair(CP_WARN,    COLOR_YELLOW, COLOR_BLACK);

    int screen_h, screen_w;
    getmaxyx(stdscr, screen_h, screen_w);

    /*
     * Layout:
     *   row  1: [sys_win 8x30] [fb_win 16x62]
     *   row 18: [cve_win fills remaining height]
     */
    WINDOW *sys_win = newwin(8,  30, 1, 1);
    WINDOW *fb_win  = newwin(16, 62, 1, 32);
    int cve_h = screen_h - 19;
    if (cve_h < 6) cve_h = 6;
    WINDOW *cve_win = newwin(cve_h, screen_w - 2, 18, 1);

    attron(COLOR_PAIR(CP_HEADER));
    mvprintw(screen_h - 1, 1,
             " q:quit  r:rescan  j/k:scroll CVEs ");
    attroff(COLOR_PAIR(CP_HEADER));
    refresh();

    int scroll_offset = 0;

    while (1) {
        update_system_info(sys_win);
        update_fabric_info(fb_win);
        update_cve_log(cve_win, scroll_offset);

        int ch = getch();
        switch (ch) {
            case 'q':
            case 'Q':
                goto cleanup;

            case 'j':
            case KEY_DOWN: {
                pthread_mutex_lock(&g_scan.lock);
                int max_scroll = g_scan.count > 0 ? g_scan.count - 1 : 0;
                pthread_mutex_unlock(&g_scan.lock);
                if (scroll_offset < max_scroll) scroll_offset++;
                break;
            }

            case 'k':
            case KEY_UP:
                if (scroll_offset > 0) scroll_offset--;
                break;

            case 'r':
            case 'R': {
                /* Only allow rescan once the previous scan has finished */
                int done;
                pthread_mutex_lock(&g_scan.lock);
                done = g_scan.scan_done;
                pthread_mutex_unlock(&g_scan.lock);
                if (done) {
                    pthread_join(scan_tid, NULL);   /* reap old thread  */
                    scroll_offset = 0;
                    pthread_mutex_lock(&g_scan.lock);
                    g_scan.scan_done = 0;
                    snprintf(g_scan.status_msg, sizeof(g_scan.status_msg), "Starting...");
                    pthread_mutex_unlock(&g_scan.lock);
                    pthread_create(&scan_tid, NULL, scanner_thread, NULL);
                }
                break;
            }

            default:
                break;
        }
    }

cleanup:
    endwin();

    pthread_join(scan_tid, NULL);   /* wait for scanner to exit cleanly */

    /* Free CVE result list */
    pthread_mutex_lock(&g_scan.lock);
    CVEResult *cur = g_scan.head;
    while (cur) { CVEResult *tmp = cur->next; free(cur); cur = tmp; }
    g_scan.head = NULL;
    pthread_mutex_unlock(&g_scan.lock);

    pthread_mutex_destroy(&g_scan.lock);
    return 0;
}
