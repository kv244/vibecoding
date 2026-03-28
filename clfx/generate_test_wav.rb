# Orchestral FX Test Burst — Sonic Pi 4.6
# Press Run — WAV is written automatically, no Record button needed.

OUT_PATH = "C:/Users/julia/MCPServer/code/vibecoding/clfx/test_orchestral_sp.wav"

set_recording_bit_depth! 16
recording_start

use_bpm 120

# 1. THE FOUNDATION: Massive orchestral hit
sample :elec_filt_snare, rate: 0.5, amp: 1.5
sample :bd_haus, amp: 2, cutoff: 70

with_fx :reverb, room: 0.8, mix: 0.6 do

  # 2. THE STRINGS: High-tension tremolo (E minor chord)
  4.times do
    synth :saw, note: :e4, release: 0.1, amp: 0.4, detune: 0.2
    synth :saw, note: :g4, release: 0.1, amp: 0.4, detune: 0.1
    synth :saw, note: :b4, release: 0.1, amp: 0.4
    sleep 0.125
  end

  # 3. THE BRASS: Sustained power chords
  synth :prophet, note: :e2, sustain: 2, release: 1, cutoff: 80, amp: 1.2
  synth :prophet, note: :e3, sustain: 2, release: 1, cutoff: 90, amp: 0.8

  # 4. THE HIGH DETAIL: Shimmering glockenspiel
  use_synth :pretty_bell
  play_pattern_timed [:e6, :g6, :b6, :e7], [0.0625], release: 0.5, amp: 0.5

end

sleep 4  # Let reverb + prophet release tails fully decay

recording_save OUT_PATH
puts "Saved: #{OUT_PATH}"
