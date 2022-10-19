from setuptools import setup, find_packages

def choose_proper_prject( requires ):
    '''
    https://stackoverflow.com/questions/14036181/
    provide-a-complex-condition-in-install-requires-python-
    setuptoolss-setup-py
    '''
    import pkg_resources
    for req in requires:
       try:
           pkg_resources.require( req )
           return [ req ]
       except pkg_resources.DistributionNotFound :
           pass
       pass
    print("There are no proper project installation available")
    print("To use this app one of the following project versions have to be installed - %s" % requires)
    import os; os._exit( os.EX_OK )
    pass


setup(name="SFcalculator_tf",
    version='0.1',
    author="Minhaun Li",
    description="A Differentiable pipeline connecting molecule models and crystallpgraphy data", 
    url=" ",
    author_email='minhuanli@g.harvard.edu',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "gemmi>=0.5.6",
        "reciprocalspaceship>=0.9.18",
        choose_proper_prject([
            "tensorflow>=2.6.0",
            "tensorflow-macos>=2.6.0"]),
        "tensorflow_probability>=0.14.0",
    ],
)