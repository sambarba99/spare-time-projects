"""
Web scraping demo with Beautiful Soup

Author: Sam Barba
Created 26/10/2022
"""

import io
import os
import requests

from bs4 import BeautifulSoup


def read_bbc_news():
	url = 'https://www.bbc.co.uk/news'
	result = requests.get(url)
	html = BeautifulSoup(result.text, 'html.parser')

	# Search for 'Most read', then for next <ol> (ordered list) tag after it in tree structure
	ordered_list = html.find(text='Most read').find_next('ol')
	list_items = ordered_list.find_all('li')  # <li> = list item

	ss = io.StringIO()
	for idx, item in enumerate(list_items):
		link_tag = item.find('a')  # <a> tags are links
		link_text = link_tag.string
		hyperlink = link_tag['href']  # Hypertext reference
		ss.write(f'\n{idx + 1}: {link_text}\n   (https://www.bbc.co.uk{hyperlink})')

	return f'Most read BBC News stories:\n{ss.getvalue()}'


def read_wiki_article_of_the_day():
	url = 'https://en.wikipedia.org/wiki/Main_Page'
	result = requests.get(url)
	html = BeautifulSoup(result.text, 'html.parser')

	# Search for "From today's featured article", then for next <p> (paragraph) tag after it in tree structure
	p = html.find(text="From today's featured article").find_next('p')
	p_text = p.text.strip().rsplit(' ', 1)[0]  # Exclude final "(Full article...)" hyperlink
	link_tag = p.find_all('a')[-1]  # Grab hyperlink manually to add to file
	hyperlink = link_tag['href']

	result = f'Wikipedia article of the day:\n\n{p_text} (https://en.wikipedia.org{hyperlink})'

	return result


def read_crypto_prices():
	url = 'https://coinmarketcap.com/'
	result = requests.get(url)
	html = BeautifulSoup(result.text, 'html.parser')

	table = html.find('table')
	table_rows = table.find_all('tr')
	headers = table_rows[0].find_all('th')[2:4]
	name_h, price_h = [h.text.strip() for h in headers]

	result = 'Crypto prices:'
	result += f'\n\n{name_h:>15}   |   {price_h}'
	result += '\n' + '-' * 38
	ss = io.StringIO()

	for tr in table_rows[1:10]:
		row_data = tr.find_all('td')[2:4]  # <td> = table data
		name, price = row_data
		ss.write(f'\n{name.find("p").text.strip():>15}   |   {price.text.strip()}')

	return f'{result}{ss.getvalue()}'


if __name__ == '__main__':
	path = os.path.expanduser('~') + '/Desktop/daily.txt'

	with open(path, 'w', encoding='UTF-8') as file:
		try:
			news_stories = read_bbc_news()
			article = read_wiki_article_of_the_day()
			prices = read_crypto_prices()
			file.write(f'{news_stories}\n\n')
			file.write(f'{article}\n\n')
			file.write(f'{prices}')
		except Exception as e:
			file.write(f'Error: {e.args[-1]}')
