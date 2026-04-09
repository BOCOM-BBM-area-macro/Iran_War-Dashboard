"""
Central configuration for project naming and metadata.

This file serves as the SINGLE SOURCE OF TRUTH for all project names and identifiers.
When renaming the project, only update this file - all other references should import from here.
"""

# ============================================================================
# PROJECT NAMING - Single Source of Truth
# ============================================================================

# Display name (user-facing, with capitalization)
PROJECT_NAME = "PeekDeck"

# Package name (Python import, lowercase with underscore)
PACKAGE_NAME = "peek_deck"

# Tagline
PROJECT_TAGLINE = "A glance is all you need"

# Description
PROJECT_DESCRIPTION = f"{PROJECT_NAME} is a configurable, widget-based monitoring system that generates static dashboards for tracking various data sources."

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# In Python code:
#     from project_config import PROJECT_NAME, PACKAGE_NAME
#     print(f"Welcome to {PROJECT_NAME}!")
#
# In templates (pass via context):
#     context = {"project_name": PROJECT_NAME, ...}
#
# In documentation:
#     <!-- Use {{ project_name }} instead of hardcoded name -->
#
# ============================================================================

# Version
__version__ = "0.1.0"
