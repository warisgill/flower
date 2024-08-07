# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Node state."""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from flwr.common import Context, RecordSet
from flwr.common.config import get_fused_config
from flwr.common.typing import Run


@dataclass()
class RunInfo:
    """Contains the Context and initial run_config of a Run."""

    context: Context
    initial_run_config: Dict[str, str]


class NodeState:
    """State of a node where client nodes execute runs."""

    def __init__(self, partition_id: Optional[int]) -> None:
        self._meta: Dict[str, Any] = {}  # holds metadata about the node
        self.run_infos: Dict[int, RunInfo] = {}
        self._partition_id = partition_id

    def register_context(
        self,
        run_id: int,
        run: Optional[Run] = None,
        flwr_dir: Optional[Path] = None,
    ) -> None:
        """Register new run context for this node."""
        if run_id not in self.run_infos:
            initial_run_config = get_fused_config(run, flwr_dir) if run else {}
            self.run_infos[run_id] = RunInfo(
                initial_run_config=initial_run_config,
                context=Context(
                    state=RecordSet(),
                    run_config=initial_run_config.copy(),
                    partition_id=self._partition_id,
                ),
            )

    def retrieve_context(self, run_id: int) -> Context:
        """Get run context given a run_id."""
        if run_id in self.run_infos:
            return self.run_infos[run_id].context

        raise RuntimeError(
            f"Context for run_id={run_id} doesn't exist."
            " A run context must be registered before it can be retrieved or updated "
            " by a client."
        )

    def update_context(self, run_id: int, context: Context) -> None:
        """Update run context."""
        if context.run_config != self.run_infos[run_id].initial_run_config:
            raise ValueError(
                "The `run_config` field of the `Context` object cannot be "
                f"modified (run_id: {run_id})."
            )
        self.run_infos[run_id].context = context
