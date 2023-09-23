import click

from quests.cli.compare import compare
from quests.cli.entropy_flex import entropy_flex
from quests.cli.entropy_sampler import entropy_sampler
from quests.cli.dentropy import dentropy


class QuestsFlexGroup(click.Group):
    pass


@click.command(cls=QuestsFlexGroup)
def quests_flex():
    """Command line interface for mkite_core"""


quests_flex.add_command(compare)
quests_flex.add_command(entropy_flex)
quests_flex.add_command(entropy_sampler)
quests_flex.add_command(dentropy)


if __name__ == "__main__":
    quests_flex()
