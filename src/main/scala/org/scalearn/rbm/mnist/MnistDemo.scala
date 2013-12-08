package org.scalearn.rbm.mnist

import org.scalearn.rbm._

object MnistDemo {
  def main(args: Array[String]) {
    val labelFile = if (args.size == 3) args(0) else "/Users/htosun/dev/dataset/train-labels-idx1-ubyte.gz"
    val featureFile = if (args.size == 3) args(1) else "/Users/htosun/dev/dataset/train-images-idx3-ubyte.gz"
    val outputDir = if (args.size == 3) args(2) else "/Users/htosun/dev/output"

    MnistBinaryDBN.start(labelFile, featureFile, outputDir)
    //MnistBinaryRBM.start(labelFile, featureFile, outputDir)

  }

}
