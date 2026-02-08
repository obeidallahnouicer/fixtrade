import os
try:
    import sqlalchemy
except Exception as e:
    print('SQLAlchemy import error:', e)
    raise

print('SCRAPING_POSTGRES_DSN=', os.environ.get('SCRAPING_POSTGRES_DSN'))
print('SQLAlchemy version:', sqlalchemy.__version__)
try:
    engine = sqlalchemy.create_engine(os.environ.get('SCRAPING_POSTGRES_DSN'))
    print('engine url:', engine.url)
    with engine.connect() as conn:
        r = conn.execute(sqlalchemy.text('SELECT current_database()'))
        print('current_database:', r.scalar())
except Exception as e:
    print('CONNECT ERROR:', e)
