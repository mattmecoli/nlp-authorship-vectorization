
import random

vocabulary = []
set(clean_books)

# We will be using a dictionary where the author is the key and their work is the value (as a list of tokenized words)



# If you want to do this for multiple texts at once, you can construct a loop as we did. That's why some of this is outside of the function (and inside a for loop) instead of inside the function itself. # Feel free to update the desire columns as you wish. Note that the function itself only adds the author attribute. If you look below the function, you'll note that we construct a dictionary of attributes (of author/work) and simply map these to the dataframe using authorname as a signal

desired_columns = ['text', 'author', 'sex', 'period']
master_paragraphs = pd.DataFrame(columns = desired_columns)

def create_paragraphs(corpus, author_name, para_size, num_para):
    start_index = random.randint(0, 50)
    end_index = start_index + para_size + 1
    jump_metric = ((len(corpus)) / (int(num_para)+5))
    jump_plus_minus = jump_metric / 10
    i = 0

    paragraphs = pd.DataFrame(columns = desired_columns)

    for x in range(int(num_para)):
        word_slice = corpus[start_index : end_index]
        string_paragraph = word_slice[0]
        for word in word_slice[1:]:
            string_paragraph = string_paragraph + ' ' + word

        paragraphs.loc[i] = [string_paragraph, author_name, None, None]

        master_paragraphs.append(paragraphs, ignore_index=True)

        jump_size = random.randint(int((jump_metric - jump_plus_minus)), int((jump_metric + jump_plus_minus)))
        start_index = random.randint(end_index, int(end_index + jump_size))
        end_index = start_index + para_size + 1
        i += 1

    return paragraphs

# ----------End of Function---------- #

para_size = 150
num_para = 100

for k, v in clean_books.items():
    paragraphs = create_paragraphs(v, k, para_size, num_para)
    master_paragraphs = pd.concat([master_paragraphs, paragraphs], ignore_index=True)

# To add the other column values you specified, use a dictionary and map

author_sex = {'KateChopin' : 'female', 'NathanielHawthorne': 'male', 'JackLondon': 'male', 'JohnLocke': 'male',
              'MargaretFuller': 'female', 'JaneAustin': 'female', 'MaryWollstonecraft': 'female',
              'VirginiaWoolf': 'female', 'MarkTwain': 'male', 'HenryDavidThoreau': 'male',
              'FScottFitzgerald': 'male', 'MaryShelley': 'female', 'EdithWharton': 'female',
              'CharlesDickens': 'male'}

work_period = {'KateChopin' : 'realism', 'NathanielHawthorne': 'gothic/romantic', 'JackLondon': 'naturalism',
               'JohnLocke': 'enlightenment', 'MargaretFuller': 'transcendentalism','JaneAustin': 'victorian',
               'MaryWollstonecraft':'enlightenment','VirginiaWoolf': 'early_modernism',
               'MarkTwain': 'realism', 'HenryDavidThoreau': 'transcendentalism',
               'FScottFitzgerald': 'early_modernism', 'MaryShelley': 'gothic/romantic',
               'EdithWharton': 'naturalism', 'CharlesDickens': 'victorian'}


master_paragraphs['sex'] = master_paragraphs['author'].map(author_sex)
master_paragraphs['period'] = master_paragraphs['author'].map(work_period)


master_paragraphs.to_csv('{}Paragraphs_{}Words.csv'.format(num_para, para_size), mode='w+')

# corpus should be in the form a normalized/tokenized list of words from doc. ['the', 'boy', 'ran']

# para_size - word count for each 'chunk' that you want. Note: unless you have manually eliminated stop_words and "infrequent words"

# num_para - how many of these paragraphs you want





# End
