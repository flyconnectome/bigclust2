# Project conventions

## Rules
- Never write to the live dataset in development or testing. The default config options for the annotation backends represent live datasets. Reading is safe, but writing is not.
- Do not leak secrets, API keys, URLs or other sensitive information in code, logs, or documentation. Use environment variables or secret management tools to handle sensitive data.