import datetime
import peewee
from ultralytics.utils import ROOT

db = peewee.SqliteDatabase(str(ROOT / "logs" / 'db.db'))

class Run(peewee.Model):
    id = peewee.CharField(primary_key=True)
    project = peewee.CharField(null=True)
    name = peewee.CharField(null=True)
    base_model = peewee.CharField(null=True)
    scale = peewee.CharField(null=True)
    data = peewee.CharField(null=True)
    group = peewee.CharField(null=True)
    notes = peewee.CharField(null=True)
    location = peewee.CharField(null=True)
    tags = peewee.CharField(null=True)
    map = peewee.CharField(default="[]")
    map50 = peewee.CharField(default="[]")
    is_basic = peewee.BooleanField(default=False)
    created = peewee.DateTimeField(default=datetime.datetime.now) 

    class Meta:
        database = db
        db_table = 'runs'


if __name__ == "__main__":
    Run.create_table()
    run1 = Run.create(id="000")
    run1.save()
