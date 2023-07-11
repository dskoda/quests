import click

from quests.cli.compare import compare


class QuestsGroup(click.Group):
    pass


@click.command(cls=QuestsGroup)
def quests():
    """Command line interface for mkite_core"""


quests.add_command(compare)


if __name__ == "__main__":
    quests()
