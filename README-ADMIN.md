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
The usage report contains the following fields in JSON format:

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
- **“sdkUsageCount”**: Number of SDK interactions a user has made.
- **“sdkUsageUpdateTs”:** Timestamp of the last update to sdkUsageCount.
