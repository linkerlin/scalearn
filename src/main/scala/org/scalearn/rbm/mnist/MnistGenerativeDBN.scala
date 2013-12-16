package org.scalearn.rbm.mnist


import java.awt.Canvas
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.awt.image.WritableRaster
import javax.swing.JFrame
import org.scalearn.rbm._
class MnistGenerativeDBN(val rbm:StackedRBM) extends Canvas{
  var sample:Layer = null
  var label = 0
 

  override def  paint(g:Graphics)= {
    rbm.synchronized {
      if(sample != null){
        val in:BufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB)
        val image:Array[Int] = sample.toArray.map(x=>Math.round(x*255f))
        val r:WritableRaster  = in.getRaster()
        r.setDataElements(0, 0, 28, 28, image);

        val  newImage:BufferedImage = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB)
             

        val g2: Graphics2D = newImage.createGraphics()
        try {
          g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,RenderingHints.VALUE_INTERPOLATION_BICUBIC)
          g2.clearRect(0, 0, 256, 256)
          g2.drawImage(in, 0, 0, 256, 256, null)
        } finally {
          g2.dispose()
        }

        g.drawImage(newImage, 10, 10, null)
        g.drawString("Generative Image for Number: "+label, 10, 300)
      }
    }
  }
  def update() = {
    val samples:List[Pair[Int, Layer]] = rbm.generateSamples(1)
    this.label = samples(0)._1
    this.sample = samples(0)._2
    println("Label:"+label+" sample(10):"+sample(10))
    repaint
  }
}
object MnistGenerativeDBN{

  def main(args: Array[String]) {
    val labelFile = if (args.size == 3) args(0) else "/Users/htosun/dev/dataset/train-labels-idx1-ubyte.gz"
    val featureFile = if (args.size == 3) args(1) else "/Users/htosun/dev/dataset/train-images-idx3-ubyte.gz"
    val outputDir = if (args.size == 3) args(2) else "/Users/htosun/dev/output"

    val DBN:MnistBinaryDBN  =  MnistBinaryDBN.train(labelFile, featureFile, outputDir)
    
    val m:MnistGenerativeDBN = new MnistGenerativeDBN(DBN.rbm)
    val frame:JFrame = new JFrame("Generative DBN")
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    frame.setSize(310, 310)


    m.setSize(310, 310)
    frame.add(m)

    frame.pack();
    frame.setLocationRelativeTo(null);
    frame.setVisible(true);

    while (true) {
      m.update();
      try {
        Thread.sleep(5000);
      } catch {
        case e:InterruptedException=>println(e)
      }
    }
    
    println("Done....")
  }
}

