def register_commands(app):
    @app.cli.command("init_db")
    def init_db_cli():
        from db import actions
        actions.init_db()

    @app.cli.command("seed_db")
    def seed_db_cli():
        from db import actions
        actions.seed_db()

    @app.cli.command("clear_db")
    def clear_db_cli():
        from db import actions
        actions.clear_db()

    @app.cli.command("drop_db")
    def drop_db_cli():
        from db import actions
        actions.drop_db()
