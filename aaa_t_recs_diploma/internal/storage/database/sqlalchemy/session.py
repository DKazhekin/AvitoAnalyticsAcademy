from sqlalchemy import Engine, create_engine


def get_db(db_url: str) -> Engine:
    engine: Engine = create_engine(db_url, echo=False, echo_pool=False)

    return engine
