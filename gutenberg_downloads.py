import time
import random

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

### NOTE: You will need to run the below three lines *first* to intialize the cache, but after the original run (which takes a couple hours), you won't need to run this code again (and will encounter an error if you do)

# from gutenberg.acquire import get_metadata_cache
# cache = get_metadata_cache()
# cache.populate()


#it's a good idea to run this in interactive mode with the functional code commented out to test your lists are the same length

book_numbers = [1245, 9830, 160, 1837, 284, 215, 11526, 205, 7370, 3420, 33, 84, 98, 1342]

author_names = ['VirginiaWoolf', 'FScottFitzgerald', 'KateChopin',
                'MarkTwain', 'EdithWharton', 'JackLondon',
                'MargaretFuller', 'HenryDavidThoreau', 'JohnLocke', 'MaryWollstonecraft', 'NathanielHawthorne', 'MaryShelley', 'CharlesDickens', 'JaneAustin']

book_names = ['NightandDay', 'TheBeautifulandDamned', 'TheAwakening', 'ThePrinceAndThePauper',
                'TheHouseOfMirth', 'TheCallOfTheWildAndWhiteFang', 'SummerOnTheLakes',
                'WaldenAndOnTheDutyOfCivilDisobedience', 'TwoTreatisesOnGovernment', 'VindicationRightsOfWomen', 'TheScarletLetter', 'Frankenstein', 'ATaleOfTwoCities', 'PrideAndPrejudice']



# Some notes about our loop. We're having the loop generate a random integer and then wait that many seconds before continuing to run. Project Gutenberg doesn't like attempts to scrape its site. The hope here is that by slowly our requests and having them occur at slightly different intervals, we'll more closely approximate a human interacting with the program. Obviously these intervals would need to be shortened for larger datasets, but they're acceptable for our 34 downloads here.


for i in range(len(author_names)):

    #get desired book for each author, write to file, wait random number of seconds to avoid suspicion
    fhand = open('{}-{}-{}'.format(i, author_names[i], book_names[i]), 'w+')
    text = strip_headers(load_etext(book_numbers[i])).strip()
    fhand.write(text)
    fhand.close()
    rand_int = random.randint(10, 30)
    print('Downloaded {}.\n Waiting {} seconds.\n\n'.format(book_names[i], rand_int))
    time.sleep(rand_int)
