"""Base widget class that all widgets inherit from."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from jinja2 import Environment, FileSystemLoader


@dataclass
class WidgetData:
    """Data returned by a widget after fetch/process/render."""

    html: str  # Rendered HTML
    data: Dict[str, Any]  # Processed data
    metadata: Dict[str, Any]  # Metadata (fetch time, source, etc.)


class BaseWidget(ABC):
    """Base class for all widgets.

    Widgets follow a 3-stage lifecycle:
    1. fetch_data() - Get raw data from external source
    2. process_data() - Transform/enrich data (optional)
    3. render() - Generate HTML from data

    Each widget instance is uniquely identified by its type + params.
    """

    def __init__(
        self,
        widget_type: str,
        params: Dict[str, Any],
        page_params: Dict[str, Any],
        update_minutes: Optional[int] = None,
    ):
        self.widget_type = widget_type
        self.params = params
        self.page_params = page_params
        self.update_minutes = update_minutes

        # Merge page params with widget params (widget params take precedence)
        self.merged_params = {**page_params, **params}

        # Set up Jinja2 environment for widget templates
        project_root = Path.cwd()
        templates_dir = project_root / "templates"
        self._jinja_env = Environment(loader=FileSystemLoader(templates_dir))

    @abstractmethod
    def fetch_data(self) -> Dict[str, Any]:
        """Fetch raw data from external source.

        Returns:
            Raw data dict to be saved to data/raw/

        Raises:
            Exception if fetch fails (will be caught and handled gracefully)
        """
        pass

    def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process/transform raw data (optional).

        Default implementation just returns raw data.
        Override for LLM processing, filtering, calculations, etc.

        Args:
            raw_data: Raw data from fetch_data()

        Returns:
            Processed data dict to be saved to data/processed/
        """
        return raw_data

    @abstractmethod
    def render(self, processed_data: Dict[str, Any]) -> str:
        """Render HTML from processed data.

        Args:
            processed_data: Processed data from process_data()

        Returns:
            HTML string for this widget
        """
        pass

    def get_required_params(self) -> list[str]:
        """Return list of required parameter names.

        Override to specify required params for validation.
        """
        return []

    def validate_params(self):
        """Validate that required params are present.

        Raises:
            ValueError if required params are missing
        """
        required = self.get_required_params()
        missing = [p for p in required if p not in self.merged_params]
        if missing:
            raise ValueError(
                f"{self.widget_type} widget missing required params: {', '.join(missing)}"
            )

    def render_template(self, template_name: str, **context) -> str:
        """Render a widget template with the given context.

        Args:
            template_name: Template path relative to templates/ (e.g., "widgets/crypto_price.html")
            **context: Template variables

        Returns:
            Rendered HTML string
        """
        template = self._jinja_env.get_template(template_name)
        return template.render(**context)
