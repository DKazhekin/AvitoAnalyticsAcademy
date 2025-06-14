import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from omegaconf import OmegaConf

from internal.app import App

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

config_path = os.getenv("CONFIG_PATH", "config.yaml")

logger.info("Load config")
config = OmegaConf.load(config_path)

logger.info("Create app")
app = App(cfg=config, logger=logger)


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    try:
        logger.info("Starting up app...")
        yield
    finally:
        logger.info("Shutting down app...")
        app.save_faiss_index()


logger.info("Create fast api app")
fast_api_app = FastAPI(lifespan=lifespan)
app.set_app(app=fast_api_app)

logger.info("Register handlers")
app.register_handlers()
