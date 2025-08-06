import logging
from opentelemetry import trace
from opentelemetry.instrumentation.logging import LoggingInstrumentor

_LOG_FORMAT = "[%(levelname)s] %(asctime)s %(name)s - %(message)s"

def init_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(format=_LOG_FORMAT, level=level)
    LoggingInstrumentor().instrument(set_logging_format=True)
    tracer = trace.get_tracer(__name__)
    logger = logging.getLogger("edge_detector")
    logger.info("Logger initialised with OTEL spans")
    return logger, tracer
