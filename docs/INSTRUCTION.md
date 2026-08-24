# Shared Agent instruction

Every harness receives the same template from [`prompt.txt`](../prompt.txt).
Only four placeholders are rendered at run initialization:

| Placeholder | Value inside the isolated run |
|---|---|
| `{base_model_path}` | `/models/base` |
| `{task_list}` | tasks from the run config |
| `{num_gpus}` | concurrent candidate limit |
| `{num_hours}` | wall-clock search budget |

No harness-specific search advice is appended. The two starter methods are
named in the prompt as optional examples; the Agent may modify or replace them.
The complete template is kept in one file so that the website, tests, and
runtime all render the same instruction.
