var convo = require('./index.js');
var assert = require("assert");


describe('1D convolution with 1d kernel', function() {
    it("should return one", function() {
        assert.deepEqual(convo.conv1d([1], [1]), [1]);
    });

    it("should return 1,2,3", function() {
        assert.deepEqual(convo.conv1d([1], [1, 2, 3]), [1, 2, 3]);
    });

    it("should return 70,140,210,280,350", function() {
        assert.deepEqual(convo.conv1d([7], [10, 20, 30, 40, 50]), [70, 140, 210, 280, 350]);
    });

    it("should return 0,0,0,0,0", function() {
        assert.deepEqual(convo.conv1d([0], [10, 20, 30, 40, 50]), [0, 0, 0, 0, 0]);
    });

    it("should return 3.92,5.46", function() {
        assert.deepEqual(convo.conv1d([1.4], [2.8, 3.9]), [3.9199999999999995, 5.46]);
    });

    it("should return 0, 2.8, 9.5, 7.8", function() {
        assert.deepEqual(convo.conv1d([0, 1, 2], [2.8, 3.9]), [0, 2.8, 9.5, 7.8]);
    });



});


describe('1D convolution with 2d kernel', function() {
    it("should return 1d array", function() {
        assert.deepEqual(convo.conv1d([1, 2, 3], [
            [1, 2, 3],
            [4, 5, 6]
        ]), [1, 4, 10, 16, 22, 28, 27, 18]);
    });

    it("should return 1d array", function() {
        assert.deepEqual(convo.conv1d([0.8, 0.7], [
            [0.2, 0.9, 0.7, 0.8],
            [0.4, 0.7, 0.3, 0.2]
        ]), [0.16000000000000003, 0.8600000000000001, 1.19, 1.1300000000000001, 0.88, 0.8399999999999999, 0.73, 0.37, 0.13999999999999999], [0.16, 0.86, 1.19, 1.13, 0.88, 0.84, 0.73, 0.37, 0.14]);
    });
});

