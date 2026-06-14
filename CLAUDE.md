# Project conventions

## Rules
- Never write to the live dataset in development or testing. The default config options for the annotation backends represent live datasets. Reading is safe, but writing is not.