from logging import Logger
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from omegaconf import DictConfig, ListConfig

from internal.adapter.database.faiss import AdapterFaissItem
from internal.adapter.database.sql import AdapterItem, AdapterUser
from internal.domain.service import ServiceItem, ServiceUser, ServiceUserItem
from internal.storage.database.faiss import create_faiss_index
from internal.storage.database.faiss.utils import (
    add_embeddings_faiss_index,
    generate_embeddings,
    load_faiss_index,
    prepare_faiss_df,
    save_faiss_index,
)
from internal.storage.database.sqlalchemy import get_db
from internal.storage.database.sqlalchemy.utils import db_from_df, prepare_df
from internal.transport.handler import HandlerItem, HandlerUser, HandlerUserItem
from src.model.collaborative import CollaborativeDataRouter


class App(object):
    def __init__(self, cfg: DictConfig | ListConfig, logger: Logger):
        self.cfg = cfg
        self.logger = logger

        self.logger.info("Set templates")
        self.templates = Jinja2Templates(directory=cfg.app.views.templates)

        self.logger.info("Get database connection")
        engine = get_db(cfg.app.database.url)
        if cfg.app.database.db_from_df:
            self.logger.info("Load data from df")
            item_df, user_df = prepare_df(
                item_df_path=cfg.app.database.item_df_path,
                user_df_path=cfg.app.database.user_df_path,
            )  # TODO: fix load big data
            db_from_df(engine=engine, item_df=item_df, user_df=user_df)

        self.logger.info("Create adapter, service, handler")
        adapter_item = AdapterItem(engine)
        if cfg.app.database.faiss.emb_from_df:
            self.logger.info("Create faiss index")
            self.faiss_index = create_faiss_index(
                embedding_dim=cfg.app.database.faiss.embedding_dim
            )
            text_column = "full_description"
            index_column = "item_id"
            self.logger.info("Prepare faiss dataframe")
            df = prepare_faiss_df(
                desc_df_path=cfg.app.database.faiss.emb_df_path, text_column=text_column
            )
            self.logger.info("Generate faiss embeddings...")
            ids, embeddings = generate_embeddings(
                dataframe=df,
                index_column=index_column,
                text_column=text_column,
                model_path=cfg.app.database.faiss.pretrained_model_path,
                model_name=cfg.app.database.faiss.pretrained_model_name,
                logger=self.logger,
            )
            self.logger.info("Add faiss embeddings...")
            add_embeddings_faiss_index(self.faiss_index, ids, embeddings)
        else:
            self.logger.info("Load faiss index...")
            self.faiss_index = load_faiss_index(
                faiss_index_path=cfg.app.database.faiss.path
            )
        adapter_faiss_item = AdapterFaissItem(faiss_index=self.faiss_index)
        service_item = ServiceItem(adapter_item, adapter_faiss_item)
        self.handler_item = HandlerItem(service_item)

        adapter_user = AdapterUser(engine)
        service_user = ServiceUser(adapter_user)
        self.handler_user = HandlerUser(service_user)

        self.logger.info("Load colaborative model...")
        cb_data_router: CollaborativeDataRouter = CollaborativeDataRouter.load(
            data_path=cfg.app.collaborative.predict,
            mapper_path=cfg.app.collaborative.mapping,
        )
        service_user_item = ServiceUserItem(service_user, service_item, cb_data_router)
        self.handler_user_item = HandlerUserItem(service_user_item)

    def set_app(self, app: FastAPI):
        self.app = app

    def register_handlers(self):
        assert self.app is not None
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        async def main(request: Request):
            ctx: Dict[str, List[Dict[str, str | int]] | Dict[str, str | int]] = dict()
            return self.templates.TemplateResponse(
                request, self.cfg.app.views.main_content_path, ctx
            )

        @app.get("/recommendation", response_class=HTMLResponse)
        def recommendation(request: Request):
            ctx: Dict[str, List[Dict[str, str | int]] | Dict[str, str | int]] = dict()
            return self.templates.TemplateResponse(
                request, self.cfg.app.views.recommendation_content_path, ctx
            )

        self.register_user_handlers(app)
        self.register_item_handlers(app)
        self.register_user_item_handlers(app)

    def register_user_handlers(self, app: FastAPI):

        @app.get("/user", response_class=JSONResponse)
        def get_user_by_id(user_id: int):
            return self.handler_user.get_user_by_id(user_id)

    def register_item_handlers(self, app: FastAPI):

        @app.get("/similar/items", response_class=JSONResponse)
        def similar_items(item_id: int):
            return self.handler_item.recommend_items_by_embedding(item_id)

        @app.get("/item", response_class=JSONResponse)
        def get_item_by_id(item_id: int):
            return self.handler_item.get_item_by_id(item_id)

        @app.get("/item/random", response_class=HTMLResponse)
        def get_random_item(request: Request):
            ctx: Dict[str, List[Dict[str, str | int]] | Dict[str, str | int]] = {}
            try:
                response = self.handler_item.get_random_item()

                ctx.update(
                    random_item=response,
                )

            except Exception as err:
                self.logger.info(err)
                ctx.update(error={"error": str(err)})
            return self.templates.TemplateResponse(
                request, self.cfg.app.views.recommendation_content_path, ctx
            )

    def register_user_item_handlers(self, app: FastAPI):

        @app.get("/user/random", response_class=HTMLResponse)
        def get_random_user(request: Request):
            ctx: Dict[str, List[Dict[str, str | int]] | Dict[str, str | int]] = {}
            try:
                response = self.handler_user_item.get_random_user_with_history()

                if isinstance(response, dict):
                    ctx.update(
                        random_user=response,
                    )
                elif isinstance(response, tuple):
                    ctx.update(
                        random_user=response[0],
                        history_items=response[1],
                    )

            except Exception as err:
                self.logger.info(err)
                ctx.update(error={"error": str(err)})
            return self.templates.TemplateResponse(
                request, self.cfg.app.views.recommendation_content_path, ctx
            )

        @app.get("/recommendation/item", response_class=HTMLResponse)
        def recommendation_item(user_id: int, request: Request):
            ctx: Dict[str, List[Dict[str, str | int]] | Dict[str, str | int]] = {}
            try:
                response = self.handler_user_item.recommendation_item(user_id)

                if isinstance(response, dict):
                    ctx.update(
                        user=response,
                    )
                elif isinstance(response, tuple):
                    ctx.update(
                        items=response[0],
                        history_items=response[1],
                    )

            except Exception as err:
                self.logger.info(err)
                ctx.update(error={"error": str(err)})
            return self.templates.TemplateResponse(
                request, self.cfg.app.views.recommendation_content_path, ctx
            )

    def save_faiss_index(self) -> None:
        save_faiss_index(self.faiss_index, self.cfg.app.database.faiss.path)
