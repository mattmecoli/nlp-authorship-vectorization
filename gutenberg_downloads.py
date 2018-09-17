import time
import random

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

### NOTE: You will need to run the below three lines *first* to intialize the cache, but after the original run (which takes a couple hours), you won't need to run this code again (and will encounter an error if you do)

# from gutenberg.acquire import get_metadata_cache
# cache = get_metadata_cache()
# cache.populate()

#it's a good idea to run this in interactive mode with the functional code commented out to test your lists are the same length

# book_numbers_1 = [7370, 3420, 33, 84, 98, 1342, 205, 11526, 215, 284, 1837, 160, 9830, 1245]

# book_numbers_2 = [10616, 134, 77, 18247, 1400, 161, 1022, 8642, 1056, 541, 86, 23810, 805, 144]

# author_names = ['JohnLocke', 'MaryWollstonecraft', 'NathanielHawthorne', 'MaryShelley', 'CharlesDickens', 'JaneAustin', 'HenryDavidThoreau', 'MargaretFuller', 'JackLondon', 'EdithWharton', 'MarkTwain', 'KateChopin', 'FScottFitzgerald', 'VirginiaWoolf']

# book_names_2 = ['LettersHumanUnderstanding', 'MariaWrongWomen',
# 'TheHouseSevenGables', 'TheLastMan', 'GreatExpectations',
# 'SenseAndSensibility', 'Walking', 'WomanInNineteenthCentury',
# 'MartinEden', 'TheAgeOfInnocence', 'ConnecticutYankee', 'AtFault',
# 'ThisSideOfParadise', 'TheVoyageOut']

#these are out of order

# book_names_1 = ['NightandDay', 'TheBeautifulandDamned', 'TheAwakening', 'ThePrinceAndThePauper',
#                 'TheHouseOfMirth', 'TheCallOfTheWildAndWhiteFang', 'SummerOnTheLakes',
#                 'WaldenAndOnTheDutyOfCivilDisobedience', 'TwoTreatisesOnGovernment', 'VindicationRightsOfWomen', 'TheScarletLetter', 'Frankenstein', 'ATaleOfTwoCities', 'PrideAndPrejudice']


extra_validation_author_names = ['AgathaChristie', 'EmilyBronte', 'GertrudeStein', 'LouisaMayAlcott', 'HarrietBeecherStowe', 'LewisCarroll', 'HermanMelville',
'JosephConrad', 'FranzKafka', 'BenjaminFranklin']

extra_validation_book_names = ['TheSecretAdversary', 'WutheringHeights', 'ThreeLives', 'LittleWomen', 'UncleTomsCabin', 'AdventuresInWonderland', 'MobyDick', 'HeartOfDarkness', 'Metamorphosis', 'Autobiography']

extra_validation_book_numbers = [1155, 768, 15408, 514, 203, 11, 2701, 219, 5200, 20203]




# Some notes about our loop. We're having the loop generate a random integer and then wait that many seconds before continuing to run. Project Gutenberg doesn't like attempts to scrape its site. The hope here is that by slowly our requests and having them occur at slightly different intervals, we'll more closely approximate a human interacting with the program. Obviously these intervals would need to be shortened for larger datasets, but they're acceptable for our 34 downloads here.


for i in range(len(extra_validation_author_names)):

    #get desired book for each author, write to file, wait random number of seconds to avoid suspicion
    fhand = open('EV-{}-{}-{}'.format((i), extra_validation_author_names[i], extra_validation_book_names[i]), 'w+')
    text = strip_headers(load_etext(extra_validation_book_numbers[i])).strip()
    fhand.write(text)
    fhand.close()
    rand_int = random.randint(10, 30)
    print('Downloaded {}.\n Waiting {} seconds.\n\n'.format(extra_validation_book_names[i], rand_int))
    time.sleep(rand_int)
