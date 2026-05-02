"""
Suppress common warnings from third-party libraries

Import this module at the top of your scripts to suppress pandas FutureWarnings
from RecBole and other common warnings.

Usage:
    import suppress_warnings  # Just import, no need to call anything
"""

import warnings

# Suppress pandas FutureWarnings from RecBole
# These are caused by RecBole's internal use of deprecated pandas patterns
# and will be fixed when RecBole updates to pandas 3.0 compatibility
warnings.filterwarnings('ignore', category=FutureWarning, module='recbole')

# Specific patterns to suppress
warnings.filterwarnings('ignore', message='.*fillna.*inplace.*')
warnings.filterwarnings('ignore', message='.*using.*len.*in Series.agg.*')
warnings.filterwarnings('ignore', message='.*chained assignment.*')

# Optional: Suppress all FutureWarnings (uncomment if needed)
# warnings.filterwarnings('ignore', category=FutureWarning)
