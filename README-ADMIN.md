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
