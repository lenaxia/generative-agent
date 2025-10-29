# Documentation Directory

This directory contains all technical documentation for the Generative Agent project. The files are organized chronologically with numeric prefixes (01-51) based on creation date.

## Documentation Files

| Filename                                                | Description                                                      |
| ------------------------------------------------------- | ---------------------------------------------------------------- |
| 01_DEPLOYMENT_GUIDE.md                                  | Guide for deploying the system                                   |
| 02_STRANDS_UNIVERSAL_AGENT_MIGRATION_PLAN.md            | Plan for migrating to the Universal Agent                        |
| 03_COMPREHENSIVE_TEST_PLAN.md                           | Comprehensive test plan                                          |
| 04_TASK_RESULT_SHARING.md                               | Documentation on task result sharing                             |
| 05_HYBRID_EXECUTION_ARCHITECTURE_DESIGN.md              | Design for hybrid execution architecture                         |
| 06_FAST_PATH_ROUTING_DESIGN.md                          | Design for fast path routing                                     |
| 07_WORKFLOW_DURATION_LOGGING.md                         | Documentation on workflow duration logging                       |
| 08_HYBRID_ROLE_LIFECYCLE_DESIGN.md                      | Design for hybrid role lifecycle                                 |
| 09_HYBRID_ROLE_MIGRATION_GUIDE.md                       | Guide for migrating to hybrid roles                              |
| 10_REDIS_SHARED_TOOLS_GUIDE.md                          | Guide for Redis shared tools                                     |
| 11_CODE_QUALITY_IMPROVEMENT_CHECKLIST.md                | Systematic checklist for code quality improvements               |
| 12_COMPREHENSIVE_TIMER_SYSTEM_DESIGN.md                 | Comprehensive timer system design and implementation             |
| 13_UNIFIED_RESULT_STORAGE_ARCHITECTURE.md               | Unified result storage architecture design                       |
| 14_LINTING_SETUP_SUMMARY.md                             | Summary of linting setup and configuration                       |
| 15_UNIFIED_COMMUNICATION_ARCHITECTURE_DESIGN.md         | Multi-channel communication architecture with background threads |
| 16_SLACK_WORKFLOW_INTEGRATION_FIX_DESIGN.md             | Design for fixing Slack message to workflow integration issue    |
| 17_TIMER_NOTIFICATION_THREADING_FIX_DESIGN.md           | Design for fixing timer notification threading issues            |
| 18_SCHEDULED_WORKFLOW_ARCHITECTURE_DESIGN.md            | Design for scheduled workflow architecture                       |
| 19_DYNAMIC_EVENT_DRIVEN_ROLE_ARCHITECTURE_DESIGN.md     | Design for dynamic event-driven role architecture                |
| 20_THREADING_ARCHITECTURE_IMPROVEMENTS.md               | Threading architecture improvements and optimizations            |
| 21_HIGH_LEVEL_ARCHITECTURE_PATTERNS.md                  | High-level architecture patterns and design principles           |
| 22_THREADING_FIX_IMPLEMENTATION_PLAN.md                 | Implementation plan for threading fixes                          |
| 23_PURE_INTENT_ARCHITECTURE_REFACTOR_DESIGN.md          | Design for pure intent architecture refactor                     |
| 24_UNIFIED_INTENT_PROCESSING_ARCHITECTURE.md            | Unified intent processing architecture design                    |
| 25_LEGACY_CODE_REMOVAL_CHECKLIST.md                     | Checklist for removing legacy code components                    |
| 26_HOME_ASSISTANT_MCP_INTEGRATION.md                    | Home Assistant MCP integration documentation                     |
| 27_TIMER_CONTEXT_ROUTING_ARCHITECTURE_DESIGN.md         | Timer context routing architecture design                        |
| 28_SINGLE_FILE_ROLE_MIGRATION_GUIDE.md                  | Guide for migrating to single file role architecture             |
| 29_ROUTER_DRIVEN_CONTEXT_SELECTION_DESIGN.md            | Router-driven context selection design                           |
| 30_FAST_REPLY_PERFORMANCE_OPTIMIZATION_DESIGN.md        | Design for fast reply performance optimization                   |
| 31_ARCHITECTURE_OVERVIEW.md                             | Overview of the system architecture                              |
| 32_API_REFERENCE.md                                     | API reference documentation                                      |
| 33_TROUBLESHOOTING_GUIDE.md                             | Guide for troubleshooting common issues                          |
| 34_CONFIGURATION_GUIDE.md                               | Guide for configuring the system                                 |
| 35_TOOL_DEVELOPMENT_GUIDE.md                            | Guide for developing new tools                                   |
| 36_NEXT_SESSION_PROMPT.md                               | Next session prompt documentation                                |
| 37_LIFECYCLE_HOOK_IMPLEMENTATION_HANDOFF.md             | Lifecycle hook implementation handoff documentation              |
| 38_CONVERSATION_ROLE_ARCHITECTURE.md                    | Conversation role architecture design                            |
| 39_PLANNING_ROLE_DESIGN.md                              | Planning role design and implementation                          |
| 40_DOCUMENT_35_TECHNICAL_DEBT_RESOLUTION.md             | Technical debt resolution for document 35                        |
| 41_DOCUMENT_35_PHASE_2_AND_3_IMPLEMENTATION_PLAN.md     | Phase 2 and 3 implementation plan for document 35                |
| 42_DOCUMENT_35_TROUBLESHOOTING_GUIDE.md                 | Troubleshooting guide for document 35                            |
| 43_INTENT_BASED_WORKFLOW_LIFECYCLE_MANAGEMENT_DESIGN.md | Intent-based workflow lifecycle management design                |
| 44_DOCUMENT_REVIEW_INCONSISTENCIES_AND_FINDINGS.md      | Document review inconsistencies and findings                     |
| 45_WORKFLOW_INTENT_CONSOLIDATION_TECHNICAL_DEBT.md      | Workflow intent consolidation technical debt                     |
| 46_PHASE_2_INTENT_DETECTION_IMPLEMENTATION.md           | Phase 2 intent detection implementation                          |
| 47_PHASE_2_TECHNICAL_DEBT_CLEANUP.md                    | Phase 2 technical debt cleanup                                   |
| 48_REMAINING_PHASE_2_ISSUES.md                          | Remaining Phase 2 issues and resolutions                         |
| 49_TECH_DEBT_CLEANUP.md                                 | Technical debt cleanup documentation                             |
| 50_TIMER_EXPIRY_FIX.md                                  | Timer expiry fix implementation                                  |
| 51_TIMER_SYSTEM_TEST_STRATEGY.md                        | Timer system test strategy and implementation                    |

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
