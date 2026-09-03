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

Generate an organization growth & adoption report as a single self-contained
HTML page, built entirely from the **admin chargeback report**. Distinct from
`usage-report` (an experiment-count PDF): `growth-report` gives an org-wide view
of workspaces, users, and platform adoption, broken down by workspace/department.

**Requires an admin API key.** The report is derived entirely from the admin
chargeback endpoint; with a non-admin key the command prints an error and exits
non-zero — there is no fallback.

### Basic Usage

```shell
cometx admin growth-report
cometx admin growth-report my-workspace
cometx admin growth-report my-workspace another-workspace
```

Chargeback data is org-wide by default. If one or more workspaces are given, the
report is scoped to just those workspaces.

### Options

- **`WORKSPACE ...`**: Zero or more workspaces to scope the report to. If omitted, the report is org-wide.
- **`--units {month,week,day,hour}`**: Chart bucket granularity (default: `month`). Charts render all-time history at this granularity — a separate concept from `--window`.
- **`--window WINDOW`**: Relative analysis window for the growth KPIs, e.g. `7d`, `14d`, `30d`, `90d`, `2w`, `6m`, `1y` (default: `7d`). Format `\d+[dwmy]`: `d`=days, `w`=weeks (×7 days), `m`=months (≈30 days), `y`=years (≈365 days).
- **`--output PATH`**: Output HTML file path (default: `growth_report.html`).
- **`--active-window WINDOW`**: Activity window for the users layer, e.g. `30d`/`60d` (default: `60d`). A user counts as *active* when their last-used timestamp falls within this window.
- **`--leaderboard-top-n N`**: Top/bottom N size for the leaderboards section (default: `5`).
- **`--exclude-personal`**: Drop workspaces whose name matches `--personal-pattern` from the chargeback data (default: off; has no effect without `--personal-pattern`).
- **`--personal-pattern REGEX`**: Regex used with `--exclude-personal` to identify personal-workspace names to drop, e.g. `'^user-'` (default: none).
- **`--no-open`**: Don't automatically open the generated HTML file after generation.
- **`--csv-dir DIR`**: Also write Glue-ready CSV fact tables (`growth_users.csv`, `growth_workspaces.csv`, `growth_org_kpis.csv`) into `DIR`. `DIR` is created if it doesn't exist.
- **`--no-html`**: Skip the HTML report. Requires `--csv-dir` (otherwise there is nothing to write, and the command errors out).
- **`--chargeback-report FILE`**: Read the chargeback report from a local JSON file (as saved by `cometx admin chargeback-report`) instead of calling the admin API. Useful for CSV-only pipelines that already have a saved chargeback snapshot, or for re-running without another API call.

### The two time concepts

- **`--units`** is the *chart granularity*: every chart shows the complete all-time history bucketed at this resolution.
- **`--window`** is the *KPI analysis window*: the "New in {window} (% of base)" growth KPIs compare accounts/workspaces created in the last `--window` against those that existed before it.

### Growth rates

The growth KPIs are computed from chargeback `createdAt` timestamps:
`new_in_window / count_before_window × 100`, where `count_before_window` is the
count that existed before `window.start` and `new_in_window` is the count created
inside `[window.start, window.end]`. Workspace creation is proxied from each
workspace's earliest member `createdAt`.

### Examples

```shell
# Org-wide report, default 7-day window
cometx admin growth-report

# 30-day window, scoped to two workspaces
cometx admin growth-report --window 30d my-workspace another-workspace

# Write to a file without auto-opening
cometx admin growth-report --no-open --output growth.html

# 30-day activity window and top/bottom-10 leaderboards, excluding personal
# workspaces named like "user-..."
cometx admin growth-report --active-window 30d --leaderboard-top-n 10 \
  --exclude-personal --personal-pattern '^user-'
```

### Output

The report generates a single self-contained HTML file containing:

- An **Organization overview (chargeback)** section with org-wide KPIs (Total workspaces, Total EM projects, New in {window} (% of base), Active workspaces %), a workspace platform-mix chart (EM / Opik / both / neither), workspace total-vs-active and added-vs-deleted charts, and a by-workspace table.
- A **Users** section with Total / Active (`--active-window`) / Active % / New in {window} KPIs, plus active-vs-total, adoption-rate, per-capability, and user-churn charts.
- A **Leaderboards** section ranking workspaces (by experiments and EM projects, exact from chargeback) and users (by Opik spans and EM activity), as top-N and active-aware bottom-N. Metrics with no data are omitted.
- A **Personal vs Service accounts** section splitting experiments / data / spans between personal and service accounts. Service accounts are identified from the admin service-accounts API when available, falling back to a labeled regex heuristic; the source is shown in the panel hint.

### Caveats

- **Chargeback is required.** The whole report is derived from the admin chargeback report; without admin access the command errors out (non-zero exit).
- **Workspace "created" is a proxy** — the earliest member `createdAt` in that workspace, since chargeback has no workspace-creation timestamp. The added-vs-deleted "deleted" series is also a best-effort proxy (all members removed) and typically reads ~0.
- **"Total projects" counts EM projects only** — chargeback's per-workspace `projects[]` covers Experiment Management. Opik projects and MPM aren't represented there (Opik appears only as a per-user span count; MPM is absent), so the platform mix uses an Opik per-user proxy and excludes MPM.
- **The people layer degrades independently.** If the chargeback payload parses but a section's inputs are missing, a warning is printed and only that section is dropped — the rest of the report still generates.

### CSV export

In addition to (or instead of) the HTML report, `growth-report` can write three
flat, Glue-ready CSV fact tables — intended for an S3 → Glue → Athena →
QuickSight pipeline. They export the underlying parsed records rather than the
HTML report's display-formatted strings, so a Glue crawler infers correct
numeric/date types instead of typing everything as `string`.

#### Flags

- **`--csv-dir DIR`**: Write the CSV files into `DIR` (created if missing). Can be combined with the normal HTML output, or with `--no-html` for CSV-only runs.
- **`--no-html`**: Skip the HTML report entirely. Requires `--csv-dir` — with neither, there is nothing to write and the command exits with an error.
- **`--chargeback-report FILE`**: Read a previously saved chargeback JSON file (from `cometx admin chargeback-report`) instead of calling the admin API. Combine with `--csv-dir` to regenerate CSVs from a snapshot without another API round-trip.

#### Files written

| File | Grain |
|---|---|
| `growth_users.csv` | one row per non-deleted user |
| `growth_workspaces.csv` | one row per workspace |
| `growth_org_kpis.csv` | one row per org-level metric (long format) |

**`growth_users.csv`**: `report_date, username, email, created_at, last_used_at, em_last_used_at, opik_last_used_at, is_suspended, is_service_account, experiment_count, data_logged_mb, opik_span_count`

**`growth_workspaces.csv`**: `report_date, workspace, member_count, num_projects, num_experiments, data_mb`

**`growth_org_kpis.csv`**: `report_date, metric_name, metric_value, metric_unit` — long format so new metrics arrive as new rows without ever changing the Glue schema. `metric_unit` is one of `count`, `percent`, `megabytes`, `label`.

#### `report_date` convention

`report_date` is the UTC date the run happened, e.g. `2026-09-03`. It is a
plain column on every row of every file — the files are flat, not written into
partition directories. On upload to S3, partition by this column, e.g.:

```
s3://your-bucket/growth-reports/report_date=2026-09-03/growth_users.csv
```

The partition keeps Athena scans cheap; the column keeps each file
self-describing on its own. A re-run on a different day produces a new
partition rather than overwriting the previous run's data.

#### Glue conventions

- lowercase `snake_case` column names
- ISO-8601 dates (`YYYY-MM-DD`), no locale formatting
- plain numbers — no thousands separators, no `%` suffixes
- booleans as `0` / `1`
- empty field (not a sentinel) for a missing value — this is what Glue reads as `NULL`
- stable column order — future additions are appended to the right, never inserted
- header row always present, even when there are zero data rows

#### Caveat: no workspace column in `growth_users.csv`

`growth_users.csv` has no `workspace` column. Chargeback reports
`experiment_count` / `data_logged_mb` / `opik_span_count` per user, not per
(user, workspace), so a user belonging to multiple workspaces appears exactly
once, with totals reported whole — this keeps `SUM(experiment_count)` over the
file correct with no `DISTINCT` handling required. The consequence is that
**per-workspace user breakdowns (e.g. "top users within workspace X") are not
answerable from `growth_users.csv` alone** — there is no user↔workspace link
in this export. Exact per-workspace totals (`member_count`, `num_projects`,
`num_experiments`, `data_mb`) live in `growth_workspaces.csv` instead.

#### Examples

```shell
# Write CSVs alongside the usual HTML report
cometx admin growth-report --csv-dir ./out

# CSV-only, no HTML
cometx admin growth-report --csv-dir ./out --no-html

# Regenerate CSVs from a saved chargeback snapshot, no API call
cometx admin growth-report --chargeback-report report.json --csv-dir ./out
```

Sample CSVs with synthetic data are available in
[`docs/csv-samples/`](docs/csv-samples/README.md).
