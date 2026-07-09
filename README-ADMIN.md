# cometx admin

To use the `cometx admin` functions, you must be in an environment with Python installed.

First, install the `cometx` Python library:

```shell
pip install cometx --upgrade
```

Next, copy your COMET_API_KEY. Login into your Comet installation, and click on your image in the upper-righthand corner, select **API Key**, and click on key to copy:

![image](https://github.com/user-attachments/assets/25d8f65b-974c-41d3-8709-4a63072d54a6)

Finally run the following:

```shell
export COMET_API_KEY=<COPY YOUR API KEY HERE>

cometx admin chargeback-report 2024-09 # for older Comet installations
cometx admin chargeback-report         # for newer Comet installations
```

## Advanced

If your installation does not support Comet Smart Keys, or your host is at an unusual location, you can also use the `--host` flag as shown:

```shell
cometx admin chargeback-report --host https://another-url.com
```

## chargeback-report

The chargeback report contains the following fields in JSON format:

- **"numberOfUsers":** total user entries in the report
- **"createdAt":** date the report was generated,
- **"organizationId":** The Comet org id

Each user entry in the report contains:

- **“username”:** The user’s Comet username.
- **“email”:** The user’s email address associated with Comet.
- **“created_at”:** The date the user was created.
- **“deletedAt”:** The date the user was deleted (for deleted users only).
- **“suspended”**: boolean flag true/false to indicate if the user has been suspended.
- **“uiUsageCount”**: Number of UI interactions a user has made.
- **“uiUsageUpdateTs”**: Timestamp of the last update to uiUsageCount.
- **"sdkUsageCount"**: Number of SDK interactions a user has made.
- **"sdkUsageUpdateTs":** Timestamp of the last update to sdkUsageCount.

## usage-report

Generate a PDF usage report with experiment counts and statistics for one or more workspaces/projects, or
start an interactive web application for dynamically creating charts and statistics.

### PDF Generation Basic Usage

```shell
cometx admin usage-report WORKSPACE
cometx admin usage-report WORKSPACE/PROJECT
cometx admin usage-report WORKSPACE1 WORKSPACE2
cometx admin usage-report WORKSPACE/PROJECT1 WORKSPACE/PROJECT2
```

### Interactive Web App

Launch an interactive Streamlit web app to select workspaces and projects from dropdown menus:

```shell
cometx admin usage-report --app
```

<img width="1638" height="839" alt="image" src="https://github.com/user-attachments/assets/abdeab5e-a138-43c9-baab-0a6aa070afb6" />


### PDF Generation Options

- **`--units {month,week,day,hour}`**: Time unit for grouping experiments (default: `month`)
  - `month`: Group by month (YYYY-MM format)
  - `week`: Group by ISO week (YYYY-WW format)
  - `day`: Group by day (YYYY-MM-DD format)
  - `hour`: Group by hour (YYYY-MM-DD-HH format)

- **`--max-experiments-per-chart N`**: Maximum number of workspaces/projects per chart (default: 100). If more workspaces/projects are provided, multiple charts will be generated.

- **`--no-open`**: Don't automatically open the generated PDF file after generation.

- **`--app`**: Launch interactive Streamlit web app instead of generating PDF.

### Examples

```shell
# Generate a report for a single workspace
cometx admin usage-report my-workspace

# Generate a report for multiple projects
cometx admin usage-report my-workspace/project1 my-workspace/project2

# Generate a report grouped by week instead of month
cometx admin usage-report workspace1 workspace2 --units week

# Generate a report grouped by day without auto-opening
cometx admin usage-report workspace --units day --no-open

# Launch interactive web app
cometx admin usage-report --app
```

### Output

The usage report generates a PDF file containing:

- **Summary statistics**: Total experiments, users, run times, GPU utilization
- **Experiment count charts**: Grouped by the specified time unit (month, week, day, or hour)
- **GPU utilization charts**: If GPU data is available for the experiments
- **GPU memory utilization charts**: If GPU data is available for the experiments

Multiple workspaces/projects are combined into a single chart with a legend. If more workspaces/projects are provided than the `--max-experiments-per-chart` limit, multiple charts will be generated.

When using the `--app` flag, an interactive web interface is launched where you can:
- Select workspace and project from dropdowns
- View statistics and charts interactively
- Change time units and regenerate reports

## growth-report

Generate a cross-platform use-case growth & adoption report — Opik projects, EM projects,
and MPM monitored models — as a single self-contained HTML page. This is distinct from
`usage-report` (an experiment-count PDF): `growth-report` tracks *how many use cases exist
and how fast they're being created*, per workspace/department, across all three products.

### Basic Usage

```shell
cometx admin growth-report
cometx admin growth-report my-workspace
cometx admin growth-report my-workspace another-workspace
```

If no workspace is given, all workspaces visible to the current API key are used.

### Options

- **`WORKSPACE ...`**: Zero or more workspaces to include. If omitted, resolved via `get_workspaces()`.
- **`--units {month,week,day,hour}`**: Chart bucket granularity (default: `month`). Charts always render **all-time** history at this granularity — this is a separate concept from `--window`, below.
- **`--window WINDOW`**: Relative analysis window for the KPI numbers, e.g. `7d`, `14d`, `30d`, `90d`, `2w`, `6m`, `1y` (default: `7d`). Format is `\d+[dwmy]`: `d`=days, `w`=weeks (×7 days), `m`=months (approximated as 30 days), `y`=years (approximated as 365 days). The month/year approximation is intentional — the window is only used to compute an "installed base before window" cutoff, not calendar-exact arithmetic.
- **`--platforms PLATFORMS`**: Comma-separated platforms to include (default: `em,opik,mpm`).
- **`--output PATH`**: Output HTML file path (default: `growth_report.html`).
- **`--limit N`**: Limit the number of workspaces processed — useful for a fast smoke-test run.
- **`--no-open`**: Don't automatically open the generated HTML file after generation.

### The two time concepts

- **`--units`** is the *chart granularity*: every chart shows the complete all-time history bucketed at this resolution.
- **`--window`** is the *KPI analysis window*: Total/New/% Growth numbers compare "in the last `--window`" against "before the window." On the charts, the window is drawn as a shaded band overlaid on the all-time series, rather than filtering the data.

### Growth rates

Growth and adoption rates are computed directly from use-case **creation timestamps**:
`pct_growth = new_in_window / count_before_window`, where `count_before_window` is the
number of use cases created before `window.start` and `new_in_window` is the number
created inside `[window.start, window.end]`.

### Examples

```shell
# Full cross-platform report for all workspaces, default 7-day window
cometx admin growth-report

# Just Opik + EM, 30-day window, for two workspaces
cometx admin growth-report --platforms em,opik --window 30d my-workspace another-workspace

# Fast smoke-test run against a single workspace, don't auto-open
cometx admin growth-report --limit 1 --no-open --output growth.html my-workspace
```

### Output

The growth report generates a single self-contained HTML file containing:

- A **unified section** ("Use cases across all platforms") with department/use-case KPIs, a stacked created-per-period chart broken down by kind (Opik/EM/MPM), a use-cases-by-department chart, and a breakdown table.
- Per-product **growth** sections (Opik / EM / MPM) with Total / New / % Growth KPIs, a new-use-cases bar chart, a cumulative area chart (both carrying the analysis window as a shaded band), and a breakdown table.
- Per-product **adoption** sections with secondary usage metrics — Opik span counts, EM experiment counts, MPM prediction volume — each broken down by project.

The breakdown tables follow one rule throughout: when more than one workspace is included,
tables break down **by workspace/department**; when only a single workspace is included,
tables break down **by individual use case** instead (a by-workspace table with one row
would be uninformative).

### Caveats

- **EM "created" is a proxy.** The Comet API has no true EM project-creation timestamp, so an EM project's creation time is approximated as the earliest experiment start time in that project, falling back to the project's `lastUpdated` time if it has no experiments.
- **Workspace/department "created" is also a proxy** — the earliest use-case creation timestamp seen in that workspace, across all platforms.
- **The EM model-registry engagement panel is a snapshot**, not an over-time series ("Registered models" / "Model versions" per workspace) — it is kept in its own labeled panel and is **never** combined with MPM's monitored-model counts, even though both are "model" concepts.
- **MPM requires provisioning.** If the account/workspace has no MPM model-monitoring provisioned, the MPM collector degrades gracefully — an empty MPM section and `collectors.mpm: false` in the underlying report data — rather than failing the whole report.
- Opik and MPM collection require their optional SDKs (`opik`, `comet_mpm`). Install them with `pip install 'cometx[all]'` if not already available; any platform whose SDK can't be imported is silently dropped from `--platforms` (a note is printed to the console) so the report still succeeds with the remaining platforms.
