from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, field
import numbers
import time
from typing import Union

from pydantic import BaseModel, Field

from bson.json_util import dumps as to_json
from bson.json_util import loads as from_json
import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy import (
    JSON,
    Float,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Index,
)
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Group(Base):
    __tablename__ = "Groups"

    _id             = Column(Integer, primary_key=True, autoincrement=True)
    name            = Column(String(256))
    namespace       = Column(String(256))
    created_time    = Column(DateTime, default=datetime.utcnow)
    meta            = Column(JSON)

    __table_args__ = (
        Index("group_name", "name"),
    )

    def to_json(self):
        return dict(
            _id=self._id,
            name=self.name,
            namespace=self.namespace,
            created_time=self.created_time,
            meta=self.meta
        )


class PyGroup(BaseModel):
    _id             : Union[int      , None] = None
    name            : str
    namespace       : Union[str      , None] = None
    created_time    : Union[datetime , None] = None
    meta            : Union[dict     , None] = Field(default_factory=dict)
    
    def to_orm(self):
        return Group(**self.dict())


class Run(Base):
    __tablename__ = "Runs"

    _id             = Column(Integer, primary_key=True, autoincrement=True)
    name            = Column(String(256))
    created_time    = Column(DateTime, default=datetime.utcnow)
    tag             = Column(JSON)
    meta            = Column(JSON)

    __table_args__ = (
        Index("pack_query", "name"),
        Index("pack_tag", "tag"),
    )

    def to_json(self):
        return dict(
            name=self.name,
            created_time=self.created_time,
            tag=self.tag,
            meta=self.meta,
        )


class PyRun(BaseModel):
    _id             : Union[int      , None] = None
    name            : str 
    created_time    : Union[datetime , None] = None
    tag             : Union[dict     , None] = None
    meta            : Union[dict     , None] = None

    def to_orm(self):
        return Run(**self.dict())


class RunGroup(Base):
    __tablename__ = "RunGroups"

    _id      = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(Integer, ForeignKey("Groups._id"), nullable=False)
    run_id   = Column(Integer, ForeignKey("Runs._id"), nullable=False)

    __table_args__ = (
        Index("group_query", "group_id", "run_id"),
    )


class PyRunGroup(BaseModel):
    _id      : Union[int      , None] = None
    group_id : int
    run_id   : int

    def to_orm(self):
        return RunGroup(**self.dict())


class Metric(Base):
    __tablename__ = "Metrics"

    _id       = Column(Integer, primary_key=True, autoincrement=True)
    run_id    = Column(Integer, ForeignKey("Runs._id"), nullable=False)

    metric    = Column(String(256))
    namespace = Column(String(256))
    time      = Column(Float)
    value     = Column(Float)
    unit      = Column(String(128))

    job_id = Column(Integer)     # Job ID
    gpu_id = Column(String(36))  # GPU id

    __table_args__ = (
        Index("metric_query", "run_id"),
        Index("metric_name", "metric"),
    )

    def to_json(self):
        return {
            'metric': self.metric,
            'time': self.time,
            'value': self.value,
            'unit': self.unit,
        }


class PyMetric(BaseModel):
    _id       : Union[int      , None] = None
    run_id    : int
    metric    : str
    namespace : Union[str      , None] = None
    time      : Union[float    , None] = None
    value     : float
    unit      : Union[str      , None] = None
    job_id    : Union[int      , None] = None
    gpu_id    : Union[str      , None] = None

    def to_orm(self):
        return Metric(**self.dict())


def generate_database_sql_setup(uri=None):
    """Users usally do not have create table permission.
    We generate the code to create the table so someone with permission can execute the script.
    """

    dummy = "sqlite:///sqlite.db"
    if uri is None:
        uri = dummy

    with open("setup.sql", "w") as file:

        def metadata_dump(sql, *multiparams, **params):
            sql = str(sql.compile(dialect=postgresql.dialect()))
            sql = sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")

            file.write(f"{sql};")
            file.write("-- \n")

        engine = sqlalchemy.create_mock_engine(
            uri, strategy="mock", executor=metadata_dump
        )
        Base.metadata.create_all(engine)


def create_database(uri):
    engine = sqlalchemy.create_engine(
        uri,
        echo=False,
        future=True,
        json_serializer=to_json,
        json_deserializer=from_json,
    )

    try:
        Base.metadata.create_all(engine)
    except DBAPIError as err:
        print("could not create database schema because of {err}")


class PySQL(BaseModel):
    sql: str


if __name__ == "__main__":
    generate_database_sql_setup()
