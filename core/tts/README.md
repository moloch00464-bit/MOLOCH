# M.O.L.O.C.H. Multi-Voice TTS System

**Status:** 🟡 PREPARATION PHASE - NOT ACTIVATED

**Phase:** 2b-prep
**Last Updated:** 2026-01-19
**Maintainer:** Markus

---

## Overview

This directory contains the **prepared architecture** for M.O.L.O.C.H.'s multi-voice Text-to-Speech (TTS) system. This is **preparation only** - no speech synthesis occurs at this stage.

### Critical Understanding

⚠️ **This system is NOT running or active**
⚠️ **No audio generation occurs**
⚠️ **No microphone integration**
⚠️ **No automatic downloads or installations**

This is an **architectural foundation** that will be explicitly activated by Markus in a future phase.

---

## Philosophy: Voices as Resources, Not Personalities

### Core Principles

1. **Dynamic Selection**: Voices are chosen contextually per utterance, not assigned to fixed personalities
2. **Context-Driven**: Selection is based on time of day, system load, interaction tone, and user preferences
3. **Local-First**: Prefer fully local TTS engines (Piper) with no cloud dependencies
4. **Transparent**: All voice selections are logged with reasoning for auditability
5. **Human-in-the-Loop**: Activation requires explicit user permission

### Why This Approach?

Traditional TTS systems often hardcode voice-personality mappings:
- "Personality A always uses Voice 1"
- "Personality B always uses Voice 2"

This is **rigid and limiting**. M.O.L.O.C.H. takes a different approach:

- Voices are **contextual resources** selected based on current needs
- Same "personality" can use different voices in different situations
- Morning briefings might need energetic voices
- Evening summaries might need calm voices
- Technical explanations might need precise voices

**Voices adapt to context, not to fixed identities.**

---

## Architecture

```
core/tts/
├── README.md                    # This file
├── tts_manager.py               # Main TTS interface (preparation only)
├── selection/
│   └── voice_selector.py        # Contextual voice selection logic
├── config/
│   ├── voices.json              # Voice metadata configuration
│   └── voices.py                # Configuration loader
└── voices/                      # Future: voice model files (.onnx)
    └── (empty - models not included)
```

### Component Responsibilities

#### `tts_manager.py`
- **Interface definition** for TTS operations
- **Prepared implementation** that logs requests without execution
- Factory function for future active implementation
- Integration point for explicit activation

#### `selection/voice_selector.py`
- **Core decision logic** for voice selection
- Scores voices based on contextual fit
- Provides transparent reasoning for selections
- Logs decisions for analysis and improvement

#### `config/voices.json`
- **Declarative configuration** of available voices
- Rich metadata: energy levels, emotional ranges, preferred contexts
- Easy to extend without code changes
- Includes installation instructions (not executed)

#### `config/voices.py`
- **Configuration loader** and validator
- Voice filtering and querying utilities
- Safe parsing with error handling

---

## Voice Selection Logic

### Input Signals

The voice selector considers:

| Signal | Description | Example Values |
|--------|-------------|----------------|
| `time_of_day` | Current time period | "morning", "afternoon", "evening", "night" |
| `system_load` | Computational resources | "low", "medium", "high" |
| `recent_interaction_tone` | Conversation sentiment | "formal", "casual", "urgent", "empathetic" |
| `explicit_user_request` | Direct voice specification | "neutral_professional", "calm_evening" |
| `session_duration` | Current session length | Integer (minutes) |
| `last_voice_used` | Previous voice for context | Voice ID string |

### Scoring Algorithm

Each voice is scored (0.0 to 1.0) based on:

1. **Time Alignment** (30% weight)
   - Morning → High-energy voices
   - Night → Calm, low-energy voices

2. **System Load Suitability** (20% weight)
   - High load → Lighter, faster voices
   - Low load → Any voice acceptable

3. **Tone Matching** (25% weight)
   - Match emotional range to conversation tone
   - Formal interactions → Professional voices
   - Casual chat → Conversational voices

4. **Diversity Bonus** (15% weight)
   - Encourage variety by avoiding recent voices
   - Prevents monotony in long sessions

5. **Continuity Bonus** (10% weight)
   - Slight preference for maintaining current voice
   - Balance between variety and consistency

**Explicit user requests always override algorithmic selection.**

### Example Selection

```
Context:
  - time_of_day: "morning"
  - system_load: "low"
  - recent_interaction_tone: "casual"
  - last_voice_used: "neutral_professional"

Result:
  voice_id: "energetic_morning"
  reason: "optimal for morning, matches casual tone, provides variety"
  confidence: 0.87
```

---

## Voice Metadata Structure

Each voice in `config/voices.json` has:

```json
{
  "voice_id": "unique_identifier",
  "display_name": "Human Readable Name",
  "engine": "piper",
  "model_path": "voices/en_US-model.onnx",
  "style_tags": ["professional", "clear"],
  "energy_level": "medium",
  "emotional_range": ["neutral", "informative"],
  "preferred_contexts": ["afternoon", "formal"],
  "language": "en-US",
  "sample_rate": 22050,
  "description": "Detailed description",
  "use_cases": ["announcements", "documentation"]
}
```

### Required Fields

- `voice_id`: Unique identifier for selection
- `style_tags`: Descriptive tags for categorization
- `energy_level`: "low", "medium", or "high"
- `emotional_range`: List of supported emotional tones
- `preferred_contexts`: Optimal usage contexts

---

## Current Voice Profiles

### 1. Neutral Professional
- **ID:** `neutral_professional`
- **Energy:** Medium
- **Best For:** Formal responses, documentation, system announcements
- **Contexts:** Afternoon, evening, formal interactions

### 2. Energetic Morning
- **ID:** `energetic_morning`
- **Energy:** High
- **Best For:** Morning greetings, task reminders, positive feedback
- **Contexts:** Morning, casual interactions

### 3. Calm Evening
- **ID:** `calm_evening`
- **Energy:** Low
- **Best For:** Evening summaries, low-energy interactions, meditation
- **Contexts:** Evening, night

### 4. Technical Precise
- **ID:** `technical_precise`
- **Energy:** Medium
- **Best For:** Code reviews, technical documentation, error reporting
- **Contexts:** Technical discussions, formal

### 5. Conversational Casual
- **ID:** `conversational_casual`
- **Energy:** Medium
- **Best For:** Casual chat, brainstorming, informal updates
- **Contexts:** Afternoon, casual interactions

---

## TTS Engine: Piper (Preferred)

### Why Piper?

- **Fully Local**: No cloud dependencies, complete privacy
- **Fast**: Optimized for real-time synthesis
- **Quality**: Neural TTS with natural prosody
- **Multi-Voice**: Extensive voice model library
- **Lightweight**: Runs on modest hardware
- **Open Source**: Transparent and auditable

### Installation (Future - Not Executed)

When activation occurs, Piper installation will involve:

1. Download Piper TTS from https://github.com/rhasspy/piper/releases
2. Download desired voice models (.onnx files)
3. Place models in `core/tts/voices/` directory
4. Update `model_path` in `config/voices.json`
5. Verify installation with test synthesis

**No automatic downloads occur. All installation is manual and explicit.**

---

## Logging Policy

### What IS Logged

✅ Voice selection decisions
✅ Selection reasoning
✅ Context snapshots (time, load, tone)
✅ Confidence scores
✅ Selection history for analysis

### What IS NOT Logged

❌ Audio data
❌ User speech content
❌ Microphone input
❌ Actual synthesized speech

**Logging is for transparency and optimization only.**

---

## Explicit Non-Goals

This system deliberately DOES NOT:

- ❌ Generate speech output (yet)
- ❌ Integrate with microphones
- ❌ Hardcode personality-voice mappings
- ❌ Automatically download voice models
- ❌ Run without user permission
- ❌ Hide decision-making process
- ❌ Require cloud services

---

## Future Activation Checklist

When Markus decides to activate this system, the following steps will be needed:

- [ ] **Permission Check**: Explicitly confirm activation with Markus
- [ ] **Install Piper TTS**: Download and set up Piper engine
- [ ] **Download Voice Models**: Select and install .onnx voice files
- [ ] **Update Configuration**: Verify `voices.json` paths are correct
- [ ] **Test Synthesis**: Verify voices work with sample text
- [ ] **Implement Active Manager**: Replace `PreparedTTSManager` with active version
- [ ] **Integration**: Connect to M.O.L.O.C.H. main system
- [ ] **Logging Verification**: Ensure logging policy is enforced
- [ ] **User Controls**: Add voice selection override commands

---

## Testing (Preparation Phase)

Even in preparation phase, you can test the decision logic:

```bash
# Test voice configuration loading
python3 core/tts/config/voices.py

# Test voice selection logic (no audio)
python3 -m core.tts.tts_manager
```

These commands:
- Load voice configurations
- Demonstrate selection logic
- Show reasoning for decisions
- **Do NOT generate audio**

---

## Integration with M.O.L.O.C.H.

### Future Integration Points

1. **System Announcements**: Use TTS for status updates
2. **Documentation Reading**: Read docs aloud (accessibility)
3. **Alerts**: Audio notifications for important events
4. **Multi-Modal Interaction**: Combine text + speech responses
5. **Accessibility**: Support for users who prefer audio

### Sample Usage (Future)

```python
from core.tts.tts_manager import create_tts_manager, ContextSignals

# Create manager (preparation mode)
tts = create_tts_manager(enabled=False)

# Test voice selection logic
context = ContextSignals(
    time_of_day="morning",
    system_load="low",
    recent_interaction_tone="casual",
    explicit_user_request=None,
    session_duration=15,
    last_voice_used=None
)

# This only logs, doesn't generate audio
selection = tts.select_voice(context)
print(f"Would use voice: {selection.voice_id}")
print(f"Reason: {selection.reason}")
```

---

## Adding Custom Voices

To add a new voice (when activated):

1. **Obtain Voice Model**: Download .onnx model from Piper repository
2. **Place Model**: Copy to `core/tts/voices/` directory
3. **Add Metadata**: Edit `config/voices.json`:

```json
{
  "voice_id": "my_custom_voice",
  "display_name": "My Custom Voice",
  "engine": "piper",
  "model_path": "voices/my_model.onnx",
  "style_tags": ["custom", "unique"],
  "energy_level": "medium",
  "emotional_range": ["neutral"],
  "preferred_contexts": ["afternoon"],
  "language": "en-US",
  "sample_rate": 22050,
  "description": "My custom voice description",
  "use_cases": ["specific use case"]
}
```

4. **Reload Configuration**: Restart or reload TTS manager

---

## Development Roadmap

### Phase 2b (Current) - Preparation ✅
- [x] Architecture design
- [x] Interface definition
- [x] Voice selection logic
- [x] Configuration structure
- [x] Logging framework
- [x] Documentation

### Phase 3 - Installation (Future)
- [ ] Piper TTS installation
- [ ] Voice model downloads
- [ ] Path verification
- [ ] Test synthesis

### Phase 4 - Activation (Future)
- [ ] Active TTS manager implementation
- [ ] System integration
- [ ] User control interface
- [ ] Accessibility features

### Phase 5 - Enhancement (Future)
- [ ] Machine learning for selection
- [ ] User feedback integration
- [ ] A/B testing framework
- [ ] Adaptive selection
- [ ] Custom voice training

---

## FAQ

**Q: Is this system currently active?**
A: No. This is preparation only. No audio generation occurs.

**Q: When will it be activated?**
A: When Markus explicitly decides to activate it in a future phase.

**Q: Can I test the voice selection logic?**
A: Yes! Run the test scripts to see selection decisions (without audio).

**Q: Are voice models included?**
A: No. Voice models must be manually downloaded when activation occurs.

**Q: What if I don't want TTS?**
A: System remains inactive unless explicitly activated. No forced audio.

**Q: How do I choose a specific voice?**
A: (Future) Set `explicit_user_request` in context, or use voice selection commands.

**Q: Can I add my own voices?**
A: (Future) Yes, by downloading models and editing `voices.json`.

**Q: Is this privacy-safe?**
A: Yes. Fully local, no cloud dependencies, no data transmission.

---

## Contact & Handoff

**Prepared By:** Claude (M.O.L.O.C.H. Assistant)
**Handoff To:** Markus
**Date:** 2026-01-19

### Handoff Notes

Markus,

This TTS system is ready for your review. Key points:

1. **No execution** - Everything is prepared but inactive
2. **Decision logic works** - You can test voice selection without audio
3. **Easy to extend** - Add voices by editing JSON
4. **Transparent** - All decisions are logged with reasoning
5. **Local-first** - No cloud dependencies required

When you're ready to activate:
- Install Piper TTS
- Download voice models
- Update paths in config
- Enable the active manager

Until then, this serves as architectural documentation and preparation.

Feel free to test the decision logic with the provided scripts!

---

## License & Attribution

- **M.O.L.O.C.H. Project**: [Your License]
- **Piper TTS**: MIT License (https://github.com/rhasspy/piper)
- **Voice Models**: Various licenses (check individual models)

---

**Remember**: This is preparation, not activation. Explicit user permission required for any speech synthesis.
