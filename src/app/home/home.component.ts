import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent {

constructor(private route: Router) {
    // this.route.navigateByUrl("home");
    }

  gotoYAI() {
    this.route.navigate(["transAI"]);
    // this.tileSelected = false;
  }

  gotoEmailAI() {
    this.route.navigateByUrl("emailAI");
    // this.tileSelected = false;
  }

  gotoFigmaAI() {
    this.route.navigateByUrl("figmaAI");
    // this.tileSelected = false;
  }

  gotoSynGenAI() {
    this.route.navigateByUrl("synGenAI");
    // this.tileSelected = false;
  }

  gotoSpeechAI() {
    this.route.navigateByUrl("speechAI");
    // this.tileSelected = false;
  }
}
