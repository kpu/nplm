#!/usr/bin/env perl

while (<>) {
    next if (/^\s*$/);
    next if (/^CANTO /);

    s/^/ /; s/$/ /;

    # punctuation
    s/([,.!?:;\(\)"]|-+)/ $1 /g;
    # lowercase
    tr/A-Z/a-z/;
    # single quotes are too much trouble

    print(join(" ", split) . "\n");
}
