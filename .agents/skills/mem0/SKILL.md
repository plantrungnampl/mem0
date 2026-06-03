```markdown
# mem0 Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the `mem0` Python codebase. You'll learn how to structure files, write imports and exports, follow commit message conventions, and organize and run tests. These patterns help ensure consistency and maintainability across the project.

## Coding Conventions

### File Naming
- Use **camelCase** for filenames.
  - Example: `myModule.py`, `dataProcessor.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import helperFunction
    ```

### Export Style
- Use **named exports** (i.e., define specific functions/classes to be imported elsewhere).
  - Example:
    ```python
    def usefulFunction():
        pass

    class ImportantClass:
        pass
    ```

### Commit Messages
- Use **conventional commit** style with prefixes.
- Prefixes detected: `fix`
- Example:
  ```
  fix: handle edge case in dataProcessor
  ```

## Workflows

### Fix a Bug
**Trigger:** When you need to correct a bug in the codebase  
**Command:** `/fix-bug`

1. Identify the bug and its location in the code.
2. Create a new branch for your fix.
3. Make the necessary code changes, following coding conventions.
4. Write or update tests (see Testing Patterns).
5. Commit your changes with a conventional commit message:
    ```
    fix: brief description of the bug fix
    ```
6. Push your branch and open a pull request.

### Add a New Module
**Trigger:** When you want to add new functionality as a separate module  
**Command:** `/add-module`

1. Create a new file using camelCase naming (e.g., `newFeature.py`).
2. Implement your functions/classes with named exports.
3. Use relative imports to access other modules.
4. Write corresponding tests in a `*.test.*` file.
5. Commit your changes with a descriptive message.
6. Push and open a pull request.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `dataProcessor.test.py`).
- The testing framework is **unknown**; check existing test files for structure.
- Place test files alongside or near the modules they test.
- Example test file:
    ```python
    # dataProcessor.test.py
    from .dataProcessor import processData

    def test_processData_valid():
        assert processData([1, 2, 3]) == [1, 4, 9]
    ```

## Commands
| Command      | Purpose                                   |
|--------------|-------------------------------------------|
| /fix-bug     | Start the workflow for fixing a bug       |
| /add-module  | Start the workflow for adding a new module|
```
