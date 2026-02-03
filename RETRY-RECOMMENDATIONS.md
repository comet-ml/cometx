# Retry configuration – environment variables

Use these variables to increase retries and backoff for experiment creation and HTTP session calls.

## Reference table

| Environment variable | Current default | Option A (moderate) | Option B (aggressive) |
|----------------------|-----------------|---------------------|------------------------|
| `COMET_GET_OR_ADD_EXPERIMENT_RETRY_TOTAL` | 10 | 25 | 100 |
| `COMET_GET_OR_ADD_EXPERIMENT_RETRY_READ` | 10 | 25 | 100 |
| `COMET_GET_OR_ADD_EXPERIMENT_RETRY_CONNECT` | 4 | 10 | 40 |
| `COMET_GET_OR_ADD_EXPERIMENT_RETRY_STATUS` | 4 | 10 | 40 |
| `COMET_GET_OR_ADD_EXPERIMENT_RETRY_BACKOFF_FACTOR` | 1 | 2 | 2 |
| `COMET_GET_OR_ADD_EXPERIMENT_RETRY_BACKOFF_MAX` | 12 | 30 | 120 |
| `COMET_HTTP_SESSION_RETRY_TOTAL` | 3 | 10 | 30 |
| `COMET_HTTP_SESSION_RETRY_BACKOFF_FACTOR` | 2 | 3 | 4 |

## Copy-paste: Option A (moderate)

```bash
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_TOTAL=25
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_READ=25
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_CONNECT=10
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_STATUS=10
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_BACKOFF_FACTOR=2
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_BACKOFF_MAX=30
export COMET_HTTP_SESSION_RETRY_TOTAL=10
export COMET_HTTP_SESSION_RETRY_BACKOFF_FACTOR=3
```

## Copy-paste: Option B (aggressive)

```bash
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_TOTAL=100
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_READ=100
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_CONNECT=40
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_STATUS=40
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_BACKOFF_FACTOR=2
export COMET_GET_OR_ADD_EXPERIMENT_RETRY_BACKOFF_MAX=120
export COMET_HTTP_SESSION_RETRY_TOTAL=30
export COMET_HTTP_SESSION_RETRY_BACKOFF_FACTOR=4
```

## Using in `.comet.config` or `.env`

Same names and values; format depends on your loader, for example:

- **.comet.config** (if your loader supports it): `COMET_GET_OR_ADD_EXPERIMENT_RETRY_TOTAL=25`
- **.env**: `COMET_GET_OR_ADD_EXPERIMENT_RETRY_TOTAL=25`
