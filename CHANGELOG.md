# Changelog

Start tracking notable changes. Our version numbers roughly follow
[Semantic Versioning](http://semver.org/) except just shifted over
a digit during the pre-1.0.0 phase. So in 0.X.Y.Z releases, X is
breaking changes, Y is new features or larger non-breaking changes, and Z is small bug fixes.
However, it is still pre-1.0 software, and does not claim to
be super stable.


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