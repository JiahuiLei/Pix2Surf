from .metric_logger import MetricLogger
from .image_logger import ImageLogger
from .model_logger import ModelLogger
from .xls_logger import XLSLogger
from .obj_logger import ObjectLogger

LOGGER_REGISTED = {
    'metric': MetricLogger,
    'image': ImageLogger,
    'model': ModelLogger,
    'xls': XLSLogger,
    'obj': ObjectLogger,
}
