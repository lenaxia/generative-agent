# Documentation Directory

This directory contains all technical documentation for the Generative Agent project. The files are organized chronologically with numeric prefixes (01-14) based on creation date.

## Documentation Files

| Filename                                         | Description                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------- |
| 01_ARCHITECTURE_OVERVIEW.md                      | Overview of the system architecture                              |
| 02_API_REFERENCE.md                              | API reference documentation                                      |
| 03_TROUBLESHOOTING_GUIDE.md                      | Guide for troubleshooting common issues                          |
| 04_CONFIGURATION_GUIDE.md                        | Guide for configuring the system                                 |
| 05_TOOL_DEVELOPMENT_GUIDE.md                     | Guide for developing new tools                                   |
| 06_STRANDS_UNIVERSAL_AGENT_MIGRATION_PLAN.md     | Plan for migrating to the Universal Agent                        |
| 07_DEPLOYMENT_GUIDE.md                           | Guide for deploying the system                                   |
| 08_COMPREHENSIVE_TEST_PLAN.md                    | Comprehensive test plan                                          |
| 09_TASK_RESULT_SHARING.md                        | Documentation on task result sharing                             |
| 10_HYBRID_EXECUTION_ARCHITECTURE_DESIGN.md       | Design for hybrid execution architecture                         |
| 11_FAST_PATH_ROUTING_DESIGN.md                   | Design for fast path routing                                     |
| 12_WORKFLOW_DURATION_LOGGING.md                  | Documentation on workflow duration logging                       |
| 13_FAST_REPLY_PERFORMANCE_OPTIMIZATION_DESIGN.md | Design for fast reply performance optimization                   |
| 14_HYBRID_ROLE_LIFECYCLE_DESIGN.md               | Design for hybrid role lifecycle                                 |
| 15_HYBRID_ROLE_MIGRATION_GUIDE.md                | Guide for migrating to hybrid roles                              |
| 16_REDIS_SHARED_TOOLS_GUIDE.md                   | Guide for Redis shared tools                                     |
| 17_CODE_QUALITY_IMPROVEMENT_CHECKLIST.md         | Systematic checklist for code quality improvements               |
| 18_COMPREHENSIVE_TIMER_SYSTEM_DESIGN.md          | Comprehensive timer system design and implementation             |
| 19_UNIFIED_RESULT_STORAGE_ARCHITECTURE.md        | Unified result storage architecture design                       |
| 20_UNIFIED_COMMUNICATION_ARCHITECTURE_DESIGN.md  | Multi-channel communication architecture with background threads |
| 21_SLACK_WORKFLOW_INTEGRATION_FIX_DESIGN.md      | Design for fixing Slack message to workflow integration issue    |

## File Organization

The numeric prefixes (01*, 02*, etc.) represent the chronological order of document creation, with 01 being the oldest document. This organization makes it easier to understand the evolution of the project's design and implementation over time.

## Usage

When adding new documentation to this directory, please follow the existing naming convention. New files should be given the next available number prefix based on their creation date.

## Rules

When creating design docs ALWAYS include a section for the following rules:

```
## Rules

- Regularly run `make lint` to validate that your code is healthy
- Always use the venv at ./venv/bin/activate
- ALWAYS use test driven development, write tests first
- Never assume tests pass, run the tests and positively verify that the test passed
- ALWAYS run all tests after making any change to ensure they are still all passing, do not move on until relevant tests are passing
- If a test fails, reflect deeply about why the test failed and fix it or fix the code
- Always write multiple tests, including happy, unhappy path and corner cases
- Always verify interfaces and data structures before writing code, do not assume the definition of a interface or data structure
- When performing refactors, ALWAYS use grep to find all instances that need to be refactored
- If you are stuck in a debugging cycle and can't seem to make forward progress, either ask for user input or take a step back and reflect on the broader scope of the code you're working on
- ALWAYS make sure your tests are meaningful, do not mock excessively, only mock where ABSOLUTELY necessary.
- Make a git commit after major changes have been completed
- When refactoring an object, refactor it in place, do not create a new file just for the sake of preserving the old version, we have git for that reason. For instance, if refactoring RequestManager, do NOT create an EnhancedRequestManager, just refactor or rewrite RequestManager
- ALWAYS Follow development and language best practices
- Use the Context7 MCP server if you need documentation for something, make sure you're looking at the right version
- Remember we are migrating AWAY from langchain TO strands agent
- Do not worry about backwards compatibility unless it is PART of a migration process and you will remove the backwards compatibility later
- Do not use fallbacks. Fallbacks tend to be brittle and fragile. Do implement fallbacks of any kind.
- Whenever you complete a phase, make sure to update this checklist
- Don't just blindly implement changes. Reflect on them to make sure they make sense within the larger project. Pull in other files if additional context is needed
- When you complete the implementation of a project add new todo items addressing outstanding technical debt related to what you just implemented, such as removing old code, updating documentation, searching for additional references, etc. Fix these issues, do not accept technical debt for the project being implemented.
```
