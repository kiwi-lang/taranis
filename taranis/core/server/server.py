from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from taranis.core.server.model import PyGroup, PyRun, PyMetric, PyRunGroup, create_database
from taranis.core.server.model import Metric, Run, Group, RunGroup


class _Server:
    def __init__(self, database="sqlite:///./test.db") -> None:
        self.app = FastAPI()

        create_database(database)

        engine = create_engine(database)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        def get_db() -> Session:
            db = SessionLocal()
            try:
                yield db
            finally:
                db.close()

        @self.app.get("/")
        def index():
            return {"message": "Hellow"}

        @self.app.post("/group/new")
        def new_group(group: PyGroup, db: Session = Depends(get_db)):
            group = group.to_orm()
            db.add(group)
            db.commit()
            db.refresh(group)
            return group._id

        @self.app.post("/group/list")
        def list_group(db: Session = Depends(get_db)):
            groups = []
            for row in db.query(Group).all():
                groups.append(row.to_json())
            return groups

        @self.app.post("/group/find/{key}={value}")
        def find_group(key:str, value:str, db: Session = Depends(get_db)):
            groups = []
            for row in db.query(Group).filter(getattr(Group, key) == value).all():
                groups.append(row.to_json())
            return groups
        
        @self.app.post("/group/upsert")
        def upsert_group(group: PyGroup, db: Session = Depends(get_db)):
            if group.meta is not None:
                key, value = list(group.meta.items())[0]

                exists = db.query(Group).filter(Group.meta[key] == str(value)).first()
     
            elif group.name is not None:
                exists = db.query(Group).filter(Group.name == group.name).first()
            
            else:
                raise RuntimeError()
            
            if exists:
                if exists.meta:
                    exists.meta.update(group.meta)
                else:
                    exists.meta = group.meta

                db.commit()
                return exists._id
           
            group = group.to_orm()
            db.add(group)
            db.commit()
            db.refresh(group) 
            return group._id

        @self.app.post("/run/new")
        def new_run(run: PyRun, db: Session = Depends(get_db)):
            run = run.to_orm()
            db.add(run)
            db.commit()
            db.refresh(run)
            return run._id
        
        @self.app.post("/run/list")
        def list_run(db: Session = Depends(get_db)):
            runs = []
            for row in db.query(Group).all():
                runs.append(row.to_json())
            return runs

        @self.app.post("/run/find/{key}={value}")
        def find_run(key:str, value:str, db: Session = Depends(get_db)):
            runs = []
            for row in db.query(Run).filter(getattr(Run, key) == value).all():
                runs.append(row.to_json())
            return runs

        @self.app.post("/group/add")
        def add_group(run_group: PyRunGroup, db: Session = Depends(get_db)):
            run_group = run_group.to_orm()

            db.add(run_group)
            db.commit()
            db.refresh(run_group)
            return run_group._id

        @self.app.post("/metric/new")
        def new_metric(metric: PyMetric, db: Session = Depends(get_db)):
            metric = metric.to_orm()

            db.add(metric)
            db.commit()
            db.refresh(metric)
            return metric._id
        
        @self.app.post("/group/fetch/runs/{group_id}")
        def group_fetch_runs(group_id: int , db: Session = Depends(get_db)):
            runs = (
                db.query(Run)
                .join(RunGroup, RunGroup.run_id == Run._id)
                .filter(RunGroup.group_id == group_id)
                .all()
            )

            data = [r.to_json() for r in runs]
            return data

        def expand_metric(metrics):
            runs = dict()
            data = []
            for metric, run in metrics:
                obj = metric.to_json()

                run_js = runs.get(run._id)
                if run_js is None:
                    run_js = run.to_json()
                    runs[run._id] = run_js

                obj.update(run_js)
                data.append(obj)

            return data
        
        @self.app.post("/group/fetch/metric/{group_id}")
        def group_fetch_metric(group_id: int , db: Session = Depends(get_db)):
            metrics = (
                db.query(Metric, Run)
                .join(RunGroup, RunGroup.run_id == Metric.run_id)
                .join(Run, Run._id == Metric.run_id)
                .filter(RunGroup.group_id == group_id)
                .all()
            )
            return expand_metric(metrics)

        @self.app.post("/run/fetch/metric/{run_id}")
        def run_fetch_metric(run_id: int, db: Session = Depends(get_db)):
            metrics = (
                db.query(Metric, Run)
                .join(Run, Run._id == Metric.run_id)
                .filter(Metric.run_id == run_id).all()
            )
            return expand_metric(metrics)

        @self.app.post("/sql/raw")
        def execute_sql(sql: PySQL, db: Session = Depends(get_db)):
            stmt = text(sql.sql)

            rows = db.execute(stmt)

            return list(rows)


def main(address="127.0.0.1", port=8000):
    my_app = _Server()

    print(f"http://{address}:{port}")

    import uvicorn
    uvicorn.run(my_app.app, host=address, port=port)


class Server:
    def __init__(self, address="127.0.0.1", port=8000) -> None:
        self.process = None
        self.address = address
        self.port = port

    @property
    def url(self):
        return f"http://{self.address}:{self.port}"

    def __enter__(self):
        import multiprocessing
        import time

        self.process = multiprocessing.Process(target=main, args=(self.address, self.port))
        self.process.start()
        time.sleep(2)

        return self

    def __exit__(self, *args):
        self.process.terminate()
        self.process.join()
        self.process.close()



if __name__ == '__main__':
    main()
