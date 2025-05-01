# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## How to Update

1. **For New Changes:**
   - Add changes under the appropriate section in `[Unreleased]`
   - Use one of these categories:
     - `Added` for new features
     - `Changed` for changes in existing functionality
     - `Fixed` for bug fixes
     - `Security` for security-related changes
   - Each change should be a single line, starting with a verb
   - Include issue/PR numbers if applicable (e.g., `(#123)`)

2. **For New Releases:**
   - Move all changes from `[Unreleased]` to a new version section
   - Update the version number and date
   - Create a new empty `[Unreleased]` section
   - Update the version in `pyproject.toml`

## [Unreleased]

### Added
- Initial implementation of TIPP (Time Invariant Portfolio Protection) strategy
- Python and Cython implementations of the TIPP model
- Type hints and documentation for better code maintainability
- Base class for TIPP implementations
- Type stubs for Cython classes

### Changed
- Improved floor calculation in TIPP strategy
- Enhanced string representation of TIPP model
- Optimized Cython implementation for better performance

### Fixed
- Fixed indentation issues in Cython code
- Corrected floor calculation formula
- Fixed type hints in Cython implementation

### Security
- No security-related changes yet

## [0.1.0] - 2024-03-21

### Added
- Initial project setup
- Basic project structure
- Dependencies configuration 