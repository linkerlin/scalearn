package org.scalearn.rbm

import scala.util._
import scala.math._

package object rbm {

  type Vec = List[Double]

  def insertAt[A, B >: A](pos: Int, b: B, lst: List[A]): List[B] = (lst, pos) match {
    case (List(), _) => b +: List()
    case (as, 0) => b +: as
    case (a +: as, i) => a +: insertAt(i - 1, b, as)
  }

}