from setuptools import setup
setup(
  name = 'dumbelek',         # How you named your package folder (MyLib)
  packages = ['dumbelek'],   # Chose the same as "name"
  version = '2.1.3',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Text processing library for languages Turkish and English',   # Give a short description about your library
  author = 'Taylan Gezici',                   # Type in your name
  author_email = 'taylangezici@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/taylangezici/dumbelek',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/taylangezici/dumbelek/archive/refs/tags/v2.1.3.tar.gz', 
  keywords = ['none'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'nltk',
          'TurkishStemmer',
          'scikit-learn',
          'pandas',
          'googletrans==4.0.0rc1'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)