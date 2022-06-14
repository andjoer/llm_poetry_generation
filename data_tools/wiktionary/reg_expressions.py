# https://de.wiktionary.org/wiki/Hilfe:Namensräume
# Pages will be ignored if title begins with:

import re

namensraum_simple = r"(Spezial|Medium|Diskussion|Vorlage|Verzeichnis|Thesaurus|Reim|Flexion|Hilfe|Kategorie|Benutzer|Gadget|Gadget-Definition|Wiktionary|Datei|MediaWiki|Modul|Benutzerin|BD|WT|Bild|Image|WikiSaurus)"
colon = r"\:"
diskussion_colon = r" Diskussion:"

''' strings to remove from xml to clean it before parsing: '''

html_comment_string = re.compile(r"(<|&lt;)!--(.(?!--(>|&gt;)))*.?--(>|&gt;)")
wiki_comment_string = re.compile(r"<comment(.(?!</comment>))*.?</comment>")
nowiki_string =  re.compile(r"(<|&lt;)nowiki(>|&gt;)(.(?!(<|&lt;)nowiki(>|&gt;)))*.?(<|&lt;)nowiki(>|&gt;)")
html_tag_string =  re.compile(r"(<|\&lt;)(small|sup|ref)(.(?!\1/\2))*.?\1\/\2(>|\&gt;)") # \2 is the second group, e.g., small, sup, ref
html_tag_string_onetag =  re.compile(r"(<|\&lt;)(small|sup|ref)(.(?!(>|\&gt;)))*.?(>|\&gt;)") # e.g., "{{Ü|en|observation}}<ref name="law_d" />,"
unicode_char_to_del = re.compile(u'[\u00AE\u200e]')

'''could also be included in to_del_strings:'''

html_break = re.compile(r"(<|\&lt;)br\s*/*(>|\&gt;)")
angle_quoted_string = re.compile(r"\(?\u00bb[^\u00ab]+\u00ab\)?")  # stuff between angle quotation marks usually comments, but these are not deleted for now.
quoted_string = re.compile(r"\(?„[^“]+“\)?")


'''list of all strings to delete:'''

to_del_strings = [html_comment_string, wiki_comment_string, nowiki_string, html_tag_string, html_tag_string_onetag, unicode_char_to_del]


'''strings to replace:'''
wiki_link_string = re.compile(r'\[\[([^\|\]]+\|)?([^\]]+)\]\]')  ## delete all hyperlink indicators: string = re.sub(wiki_link_string, r'\2', string)
specstring = re.compile("\&(amp)?;nbsp;") # e.g., "nothing&nbsp;of&nbsp;", replace as space: string = re.sub(specstring, ' ', string)
quote_html = re.compile(r'&quot;')  # replace as ": string = re.sub(quote_html, '"', string)
amp_html = re.compile(r'&amp;')  # replace as ": string = re.sub(amp_html, '&', string)

de_word_regex = re.compile(r"([^=]|^|\n)=== \{\{Wortart\W+(?:\w+\W+){1,5}?Deutsch\}\}")
title_pattern = re.compile(r"<title>(?P<pagetitle>(.(?!</title>))+.)</title>")
de_headword_spaces_allowed_regex = re.compile(r"([^=]|^|\n)== (\w(.(?!\(\{\{Sprache\|))+) \(\{\{Sprache\|Deutsch")  # headword is .group(2)
new_usage_pattern = re.compile(r"""=== {{Wortart\|([^\|]+)\|Deutsch}}""", re.U)  # A 3rd-level heading specifying word type (Wortart) and German (Deutsch) as the language
new_heading_two_regex = re.compile(r"([^=]|^|\n)== (\w\S+) \(\{\{Sprache\|(?P<lang>\w+)")
