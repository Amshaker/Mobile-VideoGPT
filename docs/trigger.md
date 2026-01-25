# Convert Mobile-VideoGPT to Streaming Inference for Real-Time Exercise Feedback

## Context & Objective

We have the **Mobile-VideoGPT** model (a video-language model) and need to modify it for **streaming inference** where it:
1. Continuously processes video frames
2. Autonomously decides **when to speak** using special action tokens
3. Generates exercise form corrections only when needed

**Inspiration**: Stream-VLM paper architecture (3D CNN + Language Model + special tokens like `<next>` and `<feedback>`)

**Current Limitation**: Mobile-VideoGPT is turn-based. We need it to process continuous video and decide when to provide feedback.

## Phase 1: Deep Analysis (Do This First)

**Before writing any code**, analyze the **Mobile-VideoGPT** repository and provide:

1. **Architecture Understanding**:
   - How does Mobile-VideoGPT currently process video? (frame-by-frame, chunks, full video?)
   - What's the model's forward pass signature?
   - How does video encoding work? (CNN, transformer, sampling strategy?)
   - What's the input format? (tensor shape, preprocessing requirements)

2. **Inference Pipeline**:
   - Where is the inference/generation code?
   - How does text generation work currently?
   - Is there existing support for: streaming, temporal context, KV-cache reuse?
   - What's the typical latency for single inference?

3. **Tokenizer & Vocabulary**:
   - What tokenizer is used?
   - Can we add special tokens easily?
   - How are tokens generated during inference?

4. **Feasibility Assessment**:
   - Can Mobile-VideoGPT's architecture support streaming without retraining?
   - What are the bottlenecks for real-time processing?
   - Are there existing hooks for temporal memory?

**Output**: A detailed markdown report answering these questions before proceeding.

## Phase 2: Design Streaming Architecture

Based on your analysis, design a streaming inference system with these components:

### Core Requirements

1. **Special Action Tokens**:
   - `<next>`: Continue observing (no speech)
   - `<feedback>`: Provide correction
   - `<correct>`: Optional positive feedback

2. **Video Stream Processing**:
   - Accept continuous frame input
   - Process in overlapping chunks (e.g., 2-second windows, 0.5s overlap)
   - Maintain sliding buffer of recent frames

3. **Temporal Context**:
   - Remember recent observations across chunks
   - Implement efficient memory management
   - Clear context after feedback or timeout

4. **Inference Logic**:
   - Predict action token first
   - Generate text only if action is `<feedback>`
   - Apply confidence thresholding
   - Enforce minimum interval between feedback (avoid spam)

### Design Constraints
- **No retraining yet**: We'll train later, so design should work with:
  - Option A: Simple rule-based action prediction (for testing infrastructure)
  - Option B: Model predictions (if we add action tokens to vocab)
- **Real-time performance**: Target >20 FPS processing
- **Modular design**: Easy to swap components and test independently

## Phase 3: Implementation

Implement the streaming system with these priorities:

### Priority 1: Core Infrastructure
```python
# 1. Video buffer for frame accumulation
# 2. Chunk extractor (overlapping windows)
# 3. Temporal context manager
# 4. Main streaming inference engine
````

### Priority 2: Model Integration

```python
# 1. Add special tokens to tokenizer
# 2. Modify forward pass if needed for streaming
# 3. Implement action token prediction (start with rule-based)
# 4. Integrate with existing Mobile-VideoGPT generation
```

### Priority 3: Demo & Testing

```python
# 1. Real-time demo script (webcam input)
# 2. Unit tests for core components
# 3. Performance profiling
```

### Code Quality Standards

* Type hints for all functions
* Comprehensive docstrings (Google style)
* Configuration-driven (YAML config file)
* Proper error handling and logging
* Clean separation of concerns

## Phase 4: Configuration & Documentation

Create:

1. **Config file** (`streaming_config.yaml`): chunk size, overlap, thresholds, generation params
2. **README**: Architecture overview, quick start, API usage
3. **Demo script**: Simple example with webcam input

## Expected Deliverables

1. **Analysis Report** (Phase 1): Mobile-VideoGPT architecture and feasibility assessment
2. **Design Document**: Proposed streaming architecture with justification
3. **Working Demo**: Real-time inference from webcam showing action prediction and feedback

## Decision Points for AI

When implementing, make informed decisions on:

1. **Action Token Strategy**:

   * Should we integrate into Mobile-VideoGPT's tokenizer now or use post-processing?
   * Rule-based vs placeholder model predictions?

2. **Temporal Context**:

   * Store hidden states, KV cache, or frame embeddings?
   * How many previous chunks to remember?

3. **Performance Optimization**:

   * Frame sampling strategy?
   * Batching opportunities?
   * Use mixed precision (fp16)?

**Document your decisions and rationale.**

## Testing Without Trained Model

Since action tokens aren't trained yet, implement a **simple rule-based policy** for testing:

```python
def rule_based_action_prediction(chunk_features, context):
    """
    Temporary logic for testing streaming infrastructure.

    Example strategies:
    - Every 5th chunk: return 'feedback'
    - Random with 20% probability: return 'feedback'
    - Detect motion change threshold: return 'feedback'

    Replace this with model prediction once trained.
    """
```

## Success Criteria

* [ ] Video stream processed continuously without blocking
* [ ] Chunks extracted with correct overlap
* [ ] Temporal context maintained across chunks
* [ ] Action tokens predicted (rule-based or model)
* [ ] Feedback generated only when appropriate
* [ ] Real-time performance achieved (>20 FPS)
* [ ] Clean, documented, modular code
* [ ] Working demo with webcam

## Important Notes

* **Start with analysis** - don't assume Mobile-VideoGPT's structure matches the prompt
* **Be flexible** - adapt design to Mobile-VideoGPT's actual architecture
* **Question assumptions** - if something seems problematic, flag it
* **Explain tradeoffs** - document why you chose specific approaches
* **Test incrementally** - validate each component before integration

Provide clear code with explanations. If you encounter ambiguities or design choices, explain your reasoning and suggest alternatives.
