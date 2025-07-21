# Changelog

Start tracking notable changes. Our version numbers roughly follow
[Semantic Versioning](http://semver.org/) except just shifted over
a digit during the pre-1.0.0 phase. So in 0.X.Y.Z releases, X is
breaking changes, Y is new features or larger non-breaking changes, and Z is small bug fixes.
However, it is still pre-1.0 software, and does not claim to
be super stable.


## [0.17.1.0]

### Added
- Exponential backoff retries for antropic requests


## [0.17.0.0]

### Changed
- Make `max_completion_tokens` removed. The attempted aliasing in 0.16.6 was
  causing some edge cases with dataclass replace. Just going to make everything
  use `max_tokens`.


## [0.16.6.0]

### Added
- reasoning_style property on a predictor

### Changed
- Make `max_tokens` and `max_completion_tokens` be direct aliases of each other
  (so we don't error out if use max_tokens on the o1 models)
- Degrade to logprob=None if requesting logprobs on a o1 model

## [0.16.5.0]

### Added
- Added Claude model names for Claude 4
- Added a __init__ in claude_wrapper so can import directly from that module

## [0.16.4.0]

### Added
- Added metadata field to LmPrompt class with generic type support. This can clean
  up usage of the batch api by associating an output with metadata. 
  It also allows potential in the future for "first available" modes for these APIs.

## [0.16.4.0]

### Added
- Added `make_reply_prompt` method to `LmPrediction` for continuing dialogs by creating new prompts that include the original conversation, the model's response, and new turns

## [0.16.3.0]

### Added
- support for duplicates in a openai batch prompt

## [0.16.2.1]

### Added
- Add model names for GPT 4.1, o4-mini, and o3.

### Changed
- The default model used when loading a open AI model to be 4.1-mini


## [0.16.1.0]

### Added
- Make openai model names and predictors picklable


## [0.16.0.0]

### Fixed
- Fixed typo of "interals.py" -> "internals.py". Breaking for the import
- Fixed a bug when could not echo chat models. Added test for it
also checking details about logprob and model internals

### Added
- Created this changelog file