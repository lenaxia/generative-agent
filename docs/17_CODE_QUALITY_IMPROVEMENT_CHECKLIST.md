# Code Quality Improvement Checklist


## Rules

* ALWAYS use test driven development, write tests first
* Never assume tests pass, run the tests and positively verify that the test passed
* ALWAYS run all tests after making any change to ensure they are still all passing, do not move on until relevant tests are passing
* If a test fails, reflect deeply about why the test failed and fix it or fix the code
* Always write multiple tests, including happy, unhappy path and corner cases
* Always verify interfaces and data structures before writing code, do not assume the definition of a interface or data structure
* When performing refactors, ALWAYS use grep to find all instances that need to be refactored
* If you are stuck in a debugging cycle and can't seem to make forward progress, either ask for user input or take a step back and reflect on the broader scope of the code you're working on
* ALWAYS make sure your tests are meaningful, do not mock excessively, only mock where ABSOLUTELY necessary.
* Make a git commit after major changes have been completed
* When refactoring an object, refactor it in place, do not create a new file just for the sake of preserving the old version, we have git for that reason. For instance, if refactoring RequestManager, do NOT create an EnhancedRequestManager, just refactor or rewrite RequestManager
* ALWAYS Follow development and language best practices
* Use the Context7 MCP server if you need documentation for something, make sure you're looking at the right version
* Remember we are migrating AWAY from langchain TO strands agent
* Do not worry about backwards compatibility unless it is PART of a migration process and you will remove the backwards compatibility later
* Do not use fallbacks
* Whenever you complete a phase, make sure to update this checklist
* Don't just blindly implement changes. Reflect on them to make sure they make sense within the larger project. Pull in other files if additional context is needed


## Overview

This document provides a systematic checklist for improving code quality in the StrandsAgent Universal Agent System. After implementing comprehensive linting tools and recent formatting fixes, we have made significant progress in code quality improvements.

## Current Status

- **✅ Linting Infrastructure**: Complete and functional
- **✅ Auto-fixes Applied**: 453 issues resolved (32% improvement from original 1,395)
- **✅ Test Compatibility**: 100% maintained (all tests passing)
- **✅ Phase 2 Critical Issues**: C901 complexity and F811 redefinitions resolved
- **✅ Formatting Issues**: Black/isort applied, 40 additional issues fixed
- **📋 Current Issues**: 828 issues remaining (down from 868 after formatting)
- **🎯 Target**: Reduce to <60 issues (93% improvement from original baseline)

## Current Issue Breakdown (828 Total Issues)

### 📊 Issue Distribution by Category

#### **Documentation Issues (D100-D415): ~520 issues (63%)**
- **D100-D104**: Missing module/package docstrings (~25 issues)
- **D101-D107**: Missing class/method docstrings (~130 issues)
- **D212**: Multi-line docstring formatting (~365 issues)
- **D415**: Missing punctuation in docstrings (~45 issues)
- **D200, D202, D205**: Other docstring formatting issues (~55 issues)

#### **Import Organization (I100, I101, I201, I202): ~180 issues (22%)**
- **I100**: Wrong import order (~90 issues)
- **I201**: Missing newlines between import groups (~85 issues)
- **I101, I202**: Import name ordering and extra newlines (~5 issues)

#### **Code Quality Issues (B007, F841, PT015, PT017, etc.): ~85 issues (10%)**
- **B007**: Unused loop variables (~25 issues)
- **F841**: Unused local variables (~15 issues)
- **PT015, PT017**: Test assertion improvements (~20 issues)
- **B011, B017, B023**: Other code quality issues (~15 issues)
- **C901**: Function complexity (1 issue in conftest.py)
- **E402, E712**: Module imports and comparisons (~10 issues)

#### **Remaining Formatting Issues: ~43 issues (5%)**
- **F401**: Unused imports (~8 issues)
- **F824**: Unused global statements (~1 issue)
- **Various**: Other minor formatting issues (~34 issues)

## Auto-Fixable Issues Checklist

### 🔧 Safe Auto-Fixes (Can be scripted - ~120 issues)

#### **High Priority - Immediate Action**

- [x] **PT009: Convert unittest assertions to pytest** (148 issues) - COMPLETED
   - **Tool**: Manual conversion using search/replace
   - **Command**: Converting `self.assertEqual()` to `assert ==` patterns
   - **Risk**: Low (pytest compatible)
   - **Files completed**: All 6 files with PT009 issues converted

- [-] **F841: Remove remaining unused variables** (~15 issues) - PARTIALLY COMPLETE
   - **Tool**: Manual removal and underscore prefixing
   - **Progress**: Fixed in `supervisor/`, some test files
   - **Risk**: Very low (safe removal)
   - **Files affected**: Various test files, need systematic review

#### **Medium Priority - Review Recommended**

- [ ] **B007: Fix loop variable naming** (~25 issues)
  - **Action**: Rename unused loop variables with underscore prefix
  - **Example**: `for item in items:` → `for _item in items:` (if item unused)
  - **Risk**: Low (naming convention)
  - **Manual review**: Required to avoid breaking used variables

- [ ] **F401: Remove unused imports** (~8 issues)
   - **Action**: Remove or comment unused imports
   - **Example**: Remove `import sys` if not used
   - **Risk**: Very low (syntax cleanup)
   - **Files**: Mainly in test files

- [x] **Formatting Issues**: Black/isort applied - 40 ISSUES FIXED
   - **Tool**: Black and isort formatters
   - **Command**: `black . && isort .`
   - **Risk**: None (formatting only)
   - **Status**: Applied to 4 files, significant improvement

## Manual Code Quality Issues Checklist

### 📝 Documentation Issues (500+ issues)

#### **D100-D104: Missing module/package docstrings** (~25 issues)

- [ ] **Add module docstrings to core modules**
  - [ ] `common/bus_packet.py` - Message bus packet definitions
  - [ ] `common/message_bus.py` - Event-driven communication system
  - [ ] `common/request_model.py` - Request data models
  - [ ] `common/task_context.py` - External state management
  - [ ] `common/task_graph.py` - Enhanced DAG with checkpointing
  - [ ] `config/__init__.py` - Configuration package
  - [ ] `config/anthropic_config.py` - Anthropic API configuration
  - [ ] `config/base_config.py` - Base configuration classes
  - [ ] `config/bedrock_config.py` - AWS Bedrock configuration
  - [ ] `config/openai_config.py` - OpenAI API configuration

#### **D101-D107: Missing class/method docstrings** (~100 issues)

- [ ] **Add class docstrings to core classes**
  - [ ] `BusPacket` class in `common/bus_packet.py`
  - [ ] `MessageBus` class in `common/message_bus.py`
  - [ ] `TaskContext` class in `common/task_context.py`
  - [ ] `TaskGraph` class in `common/task_graph.py`
  - [ ] Configuration classes in `config/` directory

- [ ] **Add method docstrings to public methods**
  - [ ] Core methods in `TaskContext` class
  - [ ] Core methods in `TaskGraph` class
  - [ ] Configuration management methods
  - [ ] Message bus methods

#### **D212: Fix docstring formatting** (~365 issues)

- [ ] **Fix multi-line docstring format**
  - **Pattern**: Move summary to first line
  - **Before**: `"""\n    Summary line\n    """`
  - **After**: `"""Summary line\n    \n    Details...\n    """`
  - **Files**: Most modules with existing docstrings

- [ ] **Add missing punctuation (D415)** (~45 issues)
  - **Action**: Add periods to docstring summaries
  - **Example**: `"""Summary line"""` → `"""Summary line."""`

### 🔒 Security & Code Quality Issues (30+ issues)

#### **High Priority - Security Issues**

- [x] **B001: Fix bare except statements** (2 issues) - COMPLETED
   - **Location**: `cli.py` lines 79, 387
   - **Action**: Replace `except:` with specific exceptions
   - **Fixed**: `except (IndexError, OSError):` and `except Exception:`
   - **Risk**: High (security vulnerability) - RESOLVED

- [x] **C901: Reduce function complexity** (1 issue) - COMPLETED
  - **Location**: `cli.py` function `run_interactive_mode` (complexity 21)
  - **Action**: Refactored into smaller helper functions
  - **Result**: All C901 issues resolved

#### **Medium Priority - Code Improvements**

- [x] **B014: Fix redundant exception types** (1 issue) - COMPLETED
   - **Location**: `tests/unit/test_config_manager_unit.py` line 210
   - **Action**: Remove `FileNotFoundError` (covered by `IOError`)

- [x] **B018: Fix useless tuple expressions** (2 issues) - COMPLETED
   - **Location**: `tests/unit/test_performance_optimizations.py`
   - **Action**: Converted to proper assertions

- [x] **C408: Fix unnecessary list calls** (1 issue) - COMPLETED
   - **Location**: `common/task_graph.py` line 256
   - **Action**: `list()` → `[]`

### 🧪 Test Quality Issues (150+ issues)

#### **PT009: Convert unittest assertions** (148 issues)

- [x] **Convert assertEqual to assert** - COMPLETED
   - **Pattern**: `self.assertEqual(a, b)` → `assert a == b`
   - **Files**: All test files converted

- [x] **Convert assertIsNone to assert** - COMPLETED
   - **Pattern**: `self.assertIsNone(x)` → `assert x is None`

- [x] **Convert assertIsInstance to assert** - COMPLETED
   - **Pattern**: `self.assertIsInstance(x, type)` → `assert isinstance(x, type)`

- [x] **Convert assertTrue to assert** - COMPLETED
   - **Pattern**: `self.assertTrue(x)` → `assert x`

#### **PT017: Fix exception assertions** (14+ issues)

- [ ] **Replace exception assertions with pytest.raises**
  - **Pattern**: `assert str(e) == "message"` → `pytest.raises(Exception, match="message")`
  - **Files**: Various test files

- [ ] **PT015: Fix assert False usage** (1 issue)
  - **Location**: `tests/unit/test_hybrid_universal_agent.py`
  - **Action**: `assert False` → `pytest.fail("message")`

### 📦 Import Organization Issues (111 issues)

#### **I100: Fix import order** (~48 issues)

- [ ] **Reorder imports by category**
  - **Order**: Standard library → Third party → First party → Local
  - **Tool**: Manual reordering (isort has limitations with complex structure)

#### **I201: Add import group separators** (~63 issues)

- [ ] **Add blank lines between import groups**
  - **Pattern**: Add blank line between different import categories
  - **Files**: Most Python files

### 🔧 Code Structure Issues

#### **E402: Module-level imports** (10 issues)

- [ ] **Move imports to top of file**
  - **Location**: `cli.py`, `tests/unit/test_redis_tools.py`
  - **Action**: Restructure code to allow top-level imports

#### **F811: Fix function redefinitions** (4 issues)

- [x] **Remove duplicate function definitions** - COMPLETED
  - **Location**: `common/task_graph.py`, `supervisor/logging_config.py`
  - **Action**: Removed duplicate `is_complete`, `mark_task_completed`, and import statements
  - **Result**: All F811 issues resolved

## Implementation Strategy

### Phase 3A: Quick Wins - Code Quality (Week 1) 🎯 TARGET: <750 issues

- [ ] **Fix unused variables (F841)** - ~15 issues
  ```bash
  # Find and fix unused variables
  flake8 . --select=F841 --show-source
  ```
- [ ] **Fix unused loop variables (B007)** - ~25 issues
  ```bash
  # Rename unused loop variables with underscore prefix
  flake8 . --select=B007 --show-source
  ```
- [ ] **Remove unused imports (F401)** - ~8 issues
  ```bash
  # Remove unused imports
  flake8 . --select=F401 --show-source
  ```
- [ ] **Fix test assertions (PT015, PT017)** - ~20 issues
  ```bash
  # Convert to proper pytest patterns
  flake8 . --select=PT015,PT017 --show-source
  ```

### Phase 3B: Import Organization (Week 2) 🎯 TARGET: <570 issues

- [ ] **Fix import ordering (I100)** - ~90 issues
  ```bash
  # Use isort with proper configuration
  isort . --check-only --diff
  isort . --force-single-line-imports
  ```
- [ ] **Add import group separators (I201)** - ~85 issues
  ```bash
  # Add blank lines between import groups
  # Manual review required for complex cases
  ```

### Phase 3C: Documentation Foundation (Weeks 3-4) 🎯 TARGET: <200 issues

- [ ] **Add missing module docstrings (D100-D104)** - ~25 issues
  - Priority files: `common/`, `config/`, `llm_provider/`
- [ ] **Add missing class docstrings (D101-D107)** - ~130 issues
  - Focus on public APIs and core classes
- [ ] **Fix docstring formatting (D212)** - ~365 issues
  ```bash
  # Systematic docstring formatting
  # Move summary to first line, add proper punctuation
  ```

### Phase 3D: Final Polish (Week 5) 🎯 TARGET: <60 issues

- [ ] **Address remaining complexity (C901)** - 1 issue in conftest.py
- [ ] **Fix remaining code quality issues** - ~15 issues
- [ ] **Final import and formatting cleanup** - ~10 issues
- [ ] **Comprehensive testing and validation**

## Verification Commands

### Before Starting Each Phase
```bash
# Backup current state
git commit -m "Before Phase X improvements"

# Verify baseline
make lint | tail -1  # Check current issue count
make test           # Verify all tests pass
```

### After Each Fix
```bash
# Verify fix didn't break anything
make test           # All tests must pass
make lint           # Check improvement
git add . && git commit -m "Fix: [issue type] - [count] issues resolved"
```

### Progress Tracking
```bash
# Check remaining issues by category
flake8 . --statistics | grep -E "^[0-9]+ +(B001|C901|D100|PT009|F841)"

# Overall progress
flake8 . --count
```

## Success Criteria

### Completion Targets

- [x] **Phase 1**: Reduce to <850 issues (automated fixes) - COMPLETED
- [x] **Phase 2**: Critical security & complexity issues - COMPLETED
- [ ] **Phase 3**: Reduce to <320 issues (documentation)
- [ ] **Phase 4**: Reduce to <170 issues (test quality)
- [ ] **Phase 5**: Reduce to <60 issues (import organization)

### Quality Gates

- [x] **All tests passing** (485+ tests)
- [x] **No security issues** (B001, B017 resolved)
- [x] **No complexity issues** (C901 resolved)
- [x] **No function redefinitions** (F811 resolved)
- [ ] **Complete documentation** (D100-D415 resolved)
- [x] **Modern test patterns** (PT009 resolved, PT017 remaining)

## Maintenance

### Ongoing Quality Assurance

- [ ] **Pre-commit hooks active** (prevents regression)
- [ ] **CI/CD pipeline functional** (automated quality checks)
- [ ] **Regular linting reviews** (weekly quality assessments)
- [ ] **Documentation updates** (keep docs current)

### Team Guidelines

- [ ] **New code standards** (follow linting rules)
- [ ] **Review process** (quality checks in PRs)
- [ ] **Training materials** (team onboarding)
- [ ] **Quality metrics** (track improvement over time)

## Resources

### Tools and Commands
- **Linting**: `make lint` - Check all issues
- **Auto-fix**: `make auto-fix-safe` - Apply safe fixes
- **Testing**: `make test` - Verify functionality
- **Formatting**: `make format` - Code formatting

### Documentation
- **Setup Guide**: `DEVELOPMENT_SETUP.md`
- **Auto-fix Guide**: `AUTO_LINTING_GUIDE.md`
- **Linting Summary**: `LINTING_SETUP_SUMMARY.md`

## Phase 3: Documentation Improvements (D100-D415)

### 📝 Module Docstrings (D100-D104) - High Priority

#### **Core Modules Missing Docstrings**
- [ ] **Add module docstrings to common/ modules**
  ```python
  """
  Common utilities and shared components for the StrandsAgent Universal Agent System.
  
  This module provides [specific functionality description].
  """
  ```
  - [ ] `common/bus_packet.py` - Message bus packet definitions
  - [ ] `common/message_bus.py` - Event-driven communication system
  - [ ] `common/request_model.py` - Request data models
  - [ ] `common/task_context.py` - External state management
  - [ ] `common/task_graph.py` - Enhanced DAG with checkpointing

- [ ] **Add module docstrings to config/ modules**
  - [ ] `config/__init__.py` - Configuration package
  - [ ] `config/anthropic_config.py` - Anthropic API configuration
  - [ ] `config/base_config.py` - Base configuration classes
  - [ ] `config/bedrock_config.py` - AWS Bedrock configuration
  - [ ] `config/openai_config.py` - OpenAI API configuration

#### **Class Docstrings (D101-D107)**
- [ ] **Add comprehensive class docstrings**
  ```python
  class TaskGraph:
      """Enhanced directed acyclic graph for task management with checkpointing.
      
      Provides task dependency management, execution tracking, and state persistence
      for complex workflow orchestration in the StrandsAgent system.
      
      Attributes:
          nodes: Dictionary mapping task IDs to TaskNode instances
          edges: List of TaskDependency objects defining relationships
          history: Execution history for audit and recovery
      """
  ```

#### **Docstring Formatting (D212, D415)**
- [ ] **Fix multi-line docstring format across codebase**
  - **Pattern**: Move summary to first line, add proper punctuation
  - **Before**: `"""\n    Summary line\n    """`
  - **After**: `"""Summary line.\n    \n    Details...\n    """`
  - **Estimated**: ~365 issues across all modules

### 🔧 Remaining Code Issues (B007, F841)

#### **Loop Variable Naming (B007)**
- [ ] **Fix unused loop variables with underscore prefix**
  ```python
  # Before
  for item in items:
      process_something()
  
  # After
  for _item in items:
      process_something()
  ```
  - **Files**: Various test files and integration modules
  - **Estimated**: ~25 issues

#### **Unused Variables (F841)**
- [ ] **Remove or prefix unused variables**
  ```python
  # Before
  result = expensive_operation()
  return success
  
  # After
  _result = expensive_operation()  # or remove if truly unused
  return success
  ```
  - **Estimated**: ~15 remaining issues

### 🧪 Test Quality Improvements (PT015, PT017, PT027)

#### **Exception Testing (PT017, PT027)**
- [ ] **Convert unittest-style exception testing to pytest**
  ```python
  # Before
  with self.assertRaises(ValueError):
      dangerous_function()
  
  # After
  with pytest.raises(ValueError, match="specific error message"):
      dangerous_function()
  ```

- [ ] **Fix assert False usage (PT015)**
  ```python
  # Before
  assert False, "This should not happen"
  
  # After
  pytest.fail("This should not happen")
  ```

### 📦 Import Organization (I100, I201)

#### **Import Ordering Strategy**
- [ ] **Standardize import order across all files**
  ```python
  # Standard library imports
  import os
  import sys
  
  # Third-party imports
  import pytest
  from pydantic import BaseModel
  
  # First-party imports
  from common.task_graph import TaskGraph
  from llm_provider.factory import LLMFactory
  ```

- [ ] **Add missing import group separators (I201)**
  - **Action**: Add blank lines between import categories
  - **Estimated**: ~63 issues across all Python files

## Phase 4: Advanced Quality Improvements

### 🔒 Security Enhancements

#### **Input Validation**
- [ ] **Add comprehensive input validation**
  - [ ] API endpoint parameter validation
  - [ ] Configuration file schema validation
  - [ ] User input sanitization in CLI

#### **Error Handling**
- [ ] **Implement structured error handling**
  - [ ] Custom exception classes with proper inheritance
  - [ ] Consistent error logging and reporting
  - [ ] Graceful degradation strategies

### ⚡ Performance Optimizations

#### **Code Efficiency**
- [ ] **Profile and optimize hot paths**
  - [ ] Identify performance bottlenecks
  - [ ] Optimize database queries and API calls
  - [ ] Implement caching strategies

#### **Memory Management**
- [ ] **Optimize memory usage**
  - [ ] Implement proper resource cleanup
  - [ ] Use generators for large data processing
  - [ ] Monitor memory leaks in long-running processes

### 🏗️ Architecture Improvements

#### **Design Patterns**
- [ ] **Implement consistent design patterns**
  - [ ] Factory pattern for component creation
  - [ ] Observer pattern for event handling
  - [ ] Strategy pattern for algorithm selection

#### **Dependency Injection**
- [ ] **Improve testability with dependency injection**
  - [ ] Interface-based design
  - [ ] Constructor injection
  - [ ] Configuration-driven dependencies

## Phase 5: Maintenance and Monitoring

### 📊 Quality Metrics

#### **Automated Quality Tracking**
- [ ] **Implement quality metrics dashboard**
  ```bash
  # Quality metrics collection
  make quality-report  # Generate comprehensive quality report
  make coverage-report # Test coverage analysis
  make complexity-report # Code complexity metrics
  ```

#### **Continuous Integration Enhancements**
- [ ] **Enhance CI/CD pipeline**
  - [ ] Quality gates in pull requests
  - [ ] Automated performance regression testing
  - [ ] Security vulnerability scanning

### 🔄 Ongoing Maintenance

#### **Regular Quality Reviews**
- [ ] **Establish quality review process**
  - [ ] Weekly linting reports
  - [ ] Monthly architecture reviews
  - [ ] Quarterly refactoring sessions

#### **Team Guidelines**
- [ ] **Create comprehensive development guidelines**
  - [ ] Code style guide
  - [ ] Testing best practices
  - [ ] Documentation standards
  - [ ] Review checklist templates

## Implementation Timeline

### Week 1-2: Documentation Foundation
- [ ] Add all missing module and class docstrings
- [ ] Fix docstring formatting issues
- [ ] Update API documentation

### Week 3-4: Code Quality Polish
- [ ] Fix remaining B007, F841 issues
- [ ] Standardize import organization
- [ ] Improve test quality (PT015, PT017, PT027)

### Week 5-6: Advanced Improvements
- [ ] Implement security enhancements
- [ ] Performance optimizations
- [ ] Architecture improvements

### Week 7-8: Maintenance Setup
- [ ] Quality metrics dashboard
- [ ] Enhanced CI/CD pipeline
- [ ] Team guidelines and processes

## Verification and Testing

### Quality Assurance Checklist
- [ ] **All linting issues resolved** (`make lint` returns 0 issues)
- [ ] **100% test coverage maintained** (`make test-coverage`)
- [ ] **Performance benchmarks met** (`make performance-test`)
- [ ] **Security scan clean** (`make security-scan`)
- [ ] **Documentation complete** (`make doc-check`)

### Automated Verification Commands
```bash
# Complete quality verification
make quality-check-all

# Individual checks
make lint                    # Linting verification
make test                   # Test suite execution
make test-coverage          # Coverage analysis
make security-scan          # Security vulnerability check
make doc-check             # Documentation completeness
make performance-test      # Performance regression test
```

---

## Appendix A: Code Quality Examples

### 📝 Documentation Best Practices

#### **Module Docstring Template**
```python
"""
StrandsAgent Universal Agent System - [Module Name]

This module provides [brief description of functionality].

Key Components:
    - [Component 1]: [Description]
    - [Component 2]: [Description]

Usage:
    Basic usage example:
    
    >>> from module import ClassName
    >>> instance = ClassName()
    >>> result = instance.method()

Dependencies:
    - [External dependency 1]
    - [External dependency 2]

Notes:
    - [Important note 1]
    - [Important note 2]

Author: StrandsAgent Development Team
Version: 1.0.0
"""
```

#### **Class Docstring Template**
```python
class TaskGraph:
    """Enhanced directed acyclic graph for task management with checkpointing.
    
    Manages task dependencies, execution state, and provides persistence
    capabilities for complex workflow orchestration. Supports both
    synchronous and asynchronous task execution patterns.
    
    Attributes:
        nodes (Dict[str, TaskNode]): Mapping of task IDs to task instances
        edges (List[TaskDependency]): Task dependency relationships
        history (List[Dict]): Execution history for audit trails
        max_history_size (int): Maximum history entries to retain
        
    Example:
        >>> graph = TaskGraph()
        >>> graph.add_task("task1", TaskDescription(...))
        >>> graph.add_dependency("task1", "task2")
        >>> ready_tasks = graph.get_ready_tasks()
        
    Raises:
        TaskGraphError: When circular dependencies are detected
        ValidationError: When task validation fails
    """
```

### 🧪 Testing Best Practices

#### **Exception Testing Best Practices**
```python
# ✅ Good: Specific exception with message matching
def test_invalid_config_raises_validation_error():
    """Test that invalid configuration raises ValidationError."""
    invalid_config = {"invalid": "config"}
    
    with pytest.raises(ValidationError, match="Missing required field"):
        ConfigManager.validate(invalid_config)

# ✅ Good: Testing exception attributes
def test_workflow_timeout_includes_details():
    """Test that timeout exception includes workflow details."""
    with pytest.raises(TimeoutError) as exc_info:
        long_running_workflow()
    
    assert exc_info.value.workflow_id == "test-workflow"
    assert exc_info.value.timeout_seconds == 30

# ❌ Avoid: Too broad exception catching
def test_bad_exception_handling():
    with pytest.raises(Exception):  # Too broad!
        some_function()
```

### 🔧 Code Quality Patterns

#### **Error Handling Patterns**
```python
# ✅ Good: Specific exception handling with context
def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load configuration from file with proper error handling."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise ConfigurationError(f"Missing config file: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        raise ConfigurationError(f"Invalid YAML syntax: {e}")
    except PermissionError:
        logger.error(f"Permission denied accessing: {config_path}")
        raise ConfigurationError(f"Cannot read config file: {config_path}")
    
    return config

# ❌ Avoid: Bare except clauses
def bad_error_handling():
    try:
        risky_operation()
    except:  # Never do this!
        pass
```

## Appendix B: Team Guidelines

### 👥 Code Review Checklist

#### **Reviewer Checklist**
- [ ] **Code Quality**
  - [ ] No linting errors introduced
  - [ ] Follows established coding standards
  - [ ] Proper error handling implemented
  - [ ] No code duplication

- [ ] **Testing**
  - [ ] New code has appropriate test coverage
  - [ ] Tests are meaningful and not just for coverage
  - [ ] Edge cases are tested
  - [ ] Tests follow naming conventions

- [ ] **Documentation**
  - [ ] Public APIs have docstrings
  - [ ] Complex logic is commented
  - [ ] README updated if needed
  - [ ] Breaking changes documented

- [ ] **Security**
  - [ ] No hardcoded secrets or credentials
  - [ ] Input validation implemented
  - [ ] SQL injection prevention
  - [ ] XSS prevention where applicable

#### **Author Checklist**
- [ ] **Before Submitting**
  - [ ] `make lint` passes without errors
  - [ ] `make test` passes all tests
  - [ ] `make test-coverage` shows adequate coverage
  - [ ] Self-review completed
  - [ ] Commit messages are descriptive

### 🎯 Definition of Done

#### **Feature Development**
- [ ] **Implementation**
  - [ ] Feature implemented according to requirements
  - [ ] Code follows team standards
  - [ ] Error handling implemented
  - [ ] Logging added where appropriate

- [ ] **Testing**
  - [ ] Unit tests written and passing
  - [ ] Integration tests added if needed
  - [ ] Manual testing completed
  - [ ] Performance impact assessed

- [ ] **Documentation**
  - [ ] Code documented with docstrings
  - [ ] API documentation updated
  - [ ] User documentation updated
  - [ ] Migration guide provided if needed

- [ ] **Quality Assurance**
  - [ ] Code review completed
  - [ ] All quality checks pass
  - [ ] Security review completed
  - [ ] Performance benchmarks met

---

## Next Immediate Actions 🚀

### Ready to Execute Now

```bash
# 1. Quick assessment of current state
flake8 . --count
make test  # Ensure all tests still pass

# 2. Fix unused variables (F841) - ~15 issues
flake8 . --select=F841 --show-source | head -20
# Then manually fix or prefix with underscore

# 3. Fix unused loop variables (B007) - ~25 issues
flake8 . --select=B007 --show-source | head -20
# Rename unused loop vars: for item in items: → for _item in items:

# 4. Remove unused imports (F401) - ~8 issues
flake8 . --select=F401 --show-source
# Remove or comment out unused imports

# 5. Fix test assertions (PT015, PT017) - ~20 issues
flake8 . --select=PT015,PT017 --show-source | head -10
# Convert assert False → pytest.fail(), fix exception assertions
```

### Expected Impact
- **Phase 3A completion**: ~68 issues fixed
- **New total**: ~760 issues (8% reduction)
- **Time estimate**: 2-3 hours of focused work
- **Risk**: Very low (safe, mechanical changes)

---

**Current Status**: 828 issues (down from 1,395 original)
**Target**: Reduce to <60 issues (93% improvement from original baseline)
**Timeline**: 4-5 weeks with systematic approach
**Success Metric**: Production-ready code quality with comprehensive documentation

**Last Updated**: January 2025 (Post-formatting fixes)
**Document Version**: 2.1
**Maintainer**: StrandsAgent Development Team