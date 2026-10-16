pragma circom 2.1.6;

include "node_modules/circomlib/circuits/comparators.circom";

template TrustScoreVerifier() {

    signal input trustScore;
    signal output valid;

    component lt = LessThan(8);

    // Compare trustScore < 75
    lt.in[0] <== trustScore;
    lt.in[1] <== 75;

    // If trustScore < 75 -> lt.out = 1
    // Therefore valid = 1 - lt.out
    valid <== 1 - lt.out;
}

component main = TrustScoreVerifier();
