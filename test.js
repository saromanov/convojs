var convo = require('./index.js');
var assert = require("assert");


describe('1D convolution with 1d kernel', function(){
    it("should return one", function(){
        assert.deepEqual(convo.conv1d([1],[1]),[1]);
    });

    it("should return 1,2,3", function(){
        assert.deepEqual(convo.conv1d([1],[1,2,3]),[1,2,3]);
    });

    it("should return 70,140,210,280,350", function(){
        assert.deepEqual(convo.conv1d([7],[10,20,30,40,50]),[70,140,210,280,350]);
    });

    it("should return 0,0,0,0,0", function(){
        assert.deepEqual(convo.conv1d([0],[10,20,30,40,50]),[0,0,0,0,0]);
    });

    it("should return 3.92,5.46", function(){
        assert.deepEqual(convo.conv1d([1.4],[2.8,3.9]),[3.9199999999999995,5.46]);
    });

    it("should return 0, 2.8, 9.5, 7.8", function(){
        assert.deepEqual(convo.conv1d([0,1,2],[2.8,3.9]), [0,2.8,9.5,7.8]);
    });



});
