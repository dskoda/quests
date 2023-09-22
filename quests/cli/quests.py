import click

from quests.cli.entropy import entropy


class QuestsGroup(click.Group):
    pass


@click.command(cls=QuestsGroup)
def quests():
    """Command line interface for quests"""


quests.add_command(entropy)


if __name__ == "__main__":
    quests()
