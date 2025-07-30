#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2023 Cometx Development
#      Team. All rights reserved.
# ****************************************

import logging
import shutil

from comet_ml.config.config_class import Config
from comet_ml.exceptions import (
    ExperimentAlreadyUploaded,
    ExperimentNotFound,
    InvalidExperimentMode,
    InvalidExperimentModeUnsupported,
)
from comet_ml.offline import OfflineSender, unzip_offline_archive

LOGGER = logging.getLogger(__name__)


# Copied from comet_ml.offline so we can have access to URL
def upload_single_offline_experiment(
    offline_archive_path: str,
    settings: "Config",
    force_upload: bool,
) -> bool:
    unzipped_directory = unzip_offline_archive(offline_archive_path)
    api_key = settings.get_string(None, "comet.api_key")
    sender = OfflineSender(
        api_key=api_key,
        offline_dir=unzipped_directory,
        force_upload=force_upload,
        display_level="info",
        override_workspace=None,
        override_project_name=None,
        message_batch_compress=settings.get_bool(
            None, "comet.message_batch.use_compression"
        ),
        message_batch_metric_interval=settings.get_int(
            None, "comet.message_batch.metric_interval"
        ),
        message_batch_metric_max_size=settings.get_int(
            None, "comet.message_batch.metric_max_size"
        ),
    )
    try:
        sender.send()
        sender.close()
        return sender._get_experiment_url()
    except ExperimentAlreadyUploaded:
        # original upload flow
        return None
    except InvalidExperimentModeUnsupported:
        # comet_ml.start() upload flow
        return None
    except InvalidExperimentMode:
        # comet_ml.start() upload flow
        return None
    except ExperimentNotFound:
        # comet_ml.start() upload flow
        return None
    finally:
        try:
            shutil.rmtree(unzipped_directory)
        except OSError:
            # We made our best effort to clean after ourselves
            msg = "Failed to clean the Offline sender tmpdir %r"
            LOGGER.debug(msg, unzipped_directory, exc_info=True)
