from taranis.core.server.model import Group, Run, Metric, RunGroup

import requests


class RemoteRun:
    def __init__(self, client, run_id) -> None:
        self.client = client
        self.run_id = run_id

    def new_metric(self, name, value, time=0, unit="", job_id=0, gpu_id="0"):
        self.client.request(
            self.client._new_metric,
            dict(
                run_id=self.run_id,
                metric=name,
                time=time,
                value=value,
                unit=unit,
                job_id=job_id,
                gpu_id=gpu_id
            )
        )

    def fetch_metrics(self):
        return self.client.request(
            self.client._run_fetch_metric + f'{self.run_id}',
            dict()
        ).json()


class RemoteGroup:
    def __init__(self, client, group_id) -> None:
        self.client = client
        self.group_id = group_id

    def add_run(self, run: RemoteRun):
        self.client.request(
            self.client._group_add,
            dict(
                group_id=self.group_id,
                run_id=run.run_id
            )
        )

    def fetch_runs(self):
        return self.client.request(
            self.client._group_fetch_runs + f'{self.group_id}',
            dict()
        ).json()
    
    def fetch_metrics(self):
        return self.client.request(
            self.client._group_fetch_metric + f'{self.group_id}',
            dict()
        ).json()


class Client:
    def __init__(self, uri) -> None:
        self.uri = uri
        
        # Group
        self._new_group = self.uri + "/group/new"
        self._upsert_group = self.uri + "/group/upsert"
        self._group_add = self.uri + "/group/add"
        self._find_group = self.uri + "/group/find/"
        self._group_fetch_metric = self.uri + '/group/fetch/metric/'
        self._group_fetch_runs = self.uri + '/group/fetch/runs/'

        # Run
        self._new_run = self.uri + "/run/new"
        self._find_run = self.uri + "/run/find/"
        self._run_fetch_metric = self.uri + '/run/fetch/metric/'

        self._new_metric = self.uri + "/metric/new"

    def new_run(self, name, tag=None, meta=None):
        response = self.request(
            self._new_run,
            dict(
                name=name,
                tag=tag,
                meta=meta,
            )
        )
        run_id = response.json()
        return RemoteRun(self, run_id)
    
    def find_group(self, name, value):
        return self.request(
            self._find_group + f'{name}={value}',
            dict()
        ).json()

    def find_run(self, name, value):
        return self.request(
            self._find_run + f'{name}={value}',
            dict()
        ).json()

    def get_group(self, name=None, meta=None):
        response = self.request(
            self._upsert_group,
            dict(
                name=name,
                meta=meta,
            )
        )
        groupe_id = response.json()
        return RemoteGroup(self, groupe_id)
    
    def new_group(self, name):
        response = self.request(
            self._new_group,
            dict(
                name=name
            )
        )
        groupe_id = response.json()
        return RemoteGroup(self, groupe_id)

    def request(self, path, data):
        return requests.post(path, json=data)

def test_server():
    from taranis.core.server.server import Server

    with Server() as server:
        client = Client(server.url)
        
        group = client.get_group("my_group2", meta=dict(digest=0))
        run = client.new_run('my_run')

        group.add_run(run)

        run.new_metric("loss", 0.10, time=0)

        print(run.fetch_metrics())
        print('GROUP:', group.fetch_runs())
        print('GROUP:', group.fetch_metrics())



if __name__ == '__main__':
    test_server()
