Regression tests for fixed bugs.

Each test subdir should contain a Makefile.am that builds a binary
called 'host' which is exected and outputs only "OK\n" (without quotes)
in case of a valid run.

