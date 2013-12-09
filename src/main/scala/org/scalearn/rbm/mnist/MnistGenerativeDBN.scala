package org.scalearn.rbm.mnist


import org.scalearn.rbm._

object MnistGenerativeDBN {
  
  def main(args: Array[String]) {
    val labelFile = if (args.size == 3) args(0) else "/Users/htosun/dev/dataset/train-labels-idx1-ubyte.gz"
    val featureFile = if (args.size == 3) args(1) else "/Users/htosun/dev/dataset/train-images-idx3-ubyte.gz"
    val outputDir = if (args.size == 3) args(2) else "/Users/htosun/dev/output"

    val DBN:MnistBinaryDBN  =  MnistBinaryDBN.train(labelFile, featureFile, outputDir)
    val samples:List[Pair[Int, Layer]] = DBN.rbm.generateSamples(10)
    for(i <- 0 until samples.size){
      val s = samples(i)  
      val image:Array[Int] = for(v <- s._2.toArray) yield Math.round(v * 255.0f)
      MnistReader.writeToJPEG("./" + s._1 + "_"+i+".jpg", DBN.dr.cols, DBN.dr.rows, image)
    }
  }
}

